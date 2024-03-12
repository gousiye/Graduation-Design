import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
import torch
from utils import ClusterDataset, FileTool, ParameterTool, Metric
from typing import Tuple
import torch.nn as nn
from models import ClusterModel
from typing import List, Optional, Sequence, Union, Any, Callable, Dict
from torch import Tensor
from torch import optim
from pre_train import PreTrain
from pytorch_lightning import Trainer
import os
from pytorch_lightning.callbacks import ModelCheckpoint
import datetime

class Train(pl.LightningModule):
    def __init__(
        self, 
        cluster_model:ClusterModel,
        cluster_dataset: ClusterDataset,
        config: dict
    )->None:
        super(Train, self).__init__()
        self.automatic_optimization = False
        self.cluster_model = cluster_model
        self.cluster_dataset = cluster_dataset
        self.config = config
        self.InitVariables()
        self.__toCuda()
        self.__GetPreModel()

    def __toCuda(self):
        for iter in self.cluster_model.encoder_decoder:
            iter.encoders.to(f'cuda:{self.devices[0]}')
            iter.decoders.to(f'cuda:{self.devices[0]}')
            iter.center.to(f'cuda:{self.devices[0]}')
        # self.cluster_dataset.dataset.H.to(f'cuda:{self.devices[0]}')

    def InitVariables(self):
        """
        把config.yaml中的参数转为类中的成员
        """
        for param_aspect in self.config:
            for parameter in self.config[param_aspect]:
                setattr(self, parameter, self.config[param_aspect][parameter])


    def __GetPreModel(self):
        """
        训练或者读取, 得到预训练的模型, 初始化AE的参数
        """
        pretrain = PreTrain(
                self.cluster_model.encoder_decoder,
                self.lr_pre,
                self.cluster_num,
                self.config
        )
        
        if self.is_pre_train == True:  
            preTrainer = Trainer(
                logger = False,
                callbacks = [ModelCheckpoint(save_last=False, save_top_k=0, monitor=None)],
                accelerator = self.pre_accelerator,
                devices = self.pre_devices,
                max_epochs = self.pre_max_epochs)
            preTrainer.fit(pretrain, self.cluster_dataset)

        pretrain.ProcessEnd(
            ParameterTool.PreDescription(self.config), 
            self.is_pre_train,
            self.save_pre_model
        )

    def forward(self):
        return self.cluster_model
    
    # 更新cluster_dataset.dataset上的H，下一轮的epoch要从这读取
    def __update_H(self, batch_idx:int) -> None:
        new_batch_h = None
        if self.cluster_model.degradation.H.is_cuda:
            new_batch_h = self.cluster_model.degradation.H.cpu()
        else:
            new_batch_h = self.cluster_model.degradation.H

        # batch_idx从0开始的
        start_idx = batch_idx * self.batch_size   
        end_idx = start_idx + new_batch_h.shape[0]
        
        self.cluster_dataset.dataset.H[start_idx: end_idx, ...] = new_batch_h.detach()


    def train_ae(self, features:Tensor) -> Tensor:
        """
        训练AE网络
        """
        opt_ae = self.optimizers()[0]
        opt_ae.zero_grad()
        z_half_list =  self.cluster_model.get_z_half_list(features)
        ae_recon_list = self.cluster_model.get_ae_recon_list(features)
        view_h_list = self.cluster_model.degradation.get_view_h_list()
        ae_recon_loss = Metric.GetAverageMSE(features, ae_recon_list)
        ae_degrade_loss = Metric.GetAverageMSE(view_h_list, z_half_list)
        ae_loss = ae_recon_loss + ae_degrade_loss
        self.manual_backward(ae_loss)
        opt_ae.step()
        return ae_loss


    def train_dg(self, features: Tensor) -> Tensor:
        """
        固定H训练degrade退化网络
        """
        # 关闭H的梯度优化
        self.cluster_model.degradation.H.requires_grad_(False)
        opt_dg = self.optimizers()[1]
        opt_dg.zero_grad()
        # train_ae更新ae网络后才调用，需要重新计算
        z_half_list =  self.cluster_model.get_z_half_list(features)
        view_h_list = self.cluster_model.degradation.get_view_h_list()
        dg_loss = Metric.GetAverageMSE(z_half_list, view_h_list)
        self.manual_backward(dg_loss)
        opt_dg.step()
        return dg_loss

    def train_h(self, features:Tensor) -> Tensor:
        """
        训练H
        """
        # 打开H的梯度优化
        self.cluster_model.degradation.H.requires_grad_(True)
        opt_h = (optim.Adam(self.cluster_model.degradation.parameters(), lr = self.lr_h))
        h_loss_avg = 0
        # 单独训练H
        for i in range(self.H_max_epochs):
            opt_h.zero_grad()        
            # dg, ae网络的参数有变化，因此需要重新计算
            view_h_list = self.cluster_model.degradation.get_view_h_list()
            z_half_list =  self.cluster_model.get_z_half_list(features)
            h_loss = Metric.GetAverageMSE(z_half_list, view_h_list)
            h_loss_avg += h_loss
            self.manual_backward(h_loss)
            opt_h.step()
        h_loss_avg /= self.H_max_epochs
        return h_loss_avg 


    def training_step(self, batch: Tuple, batch_idx):
        features, labels, h = batch
        self.cluster_model.degradation.set_H(h)
        # print()
        # print(id(h.data))
        # print(id(self.cluster_model.degradation.H.data))
        # print('----------------------------')
        ae_loss = self.train_ae(features)
        dg_loss = self.train_dg(features)
        h_loss = self.train_h(features)
        self.log('ae_loss', ae_loss)
        self.log('dg_loss', dg_loss)
        self.log('h_loss', h_loss)
        self.__update_H(batch_idx)
        return {'loss':h_loss, 'ae_loss':ae_loss, 'dg_loss':dg_loss, 'h_loss':h_loss}
    

    def training_epoch_end(self, outputs: List) -> None:
        print()
        print(outputs[0])
        # print((self.cluster_model.degradation.H.shape))
        # for group in self.optimizers()[1].param_groups:
        #     for param in group['params']:
        #         print(id(param))
        # print('----------------------------------------------------')
        
        # for group in opt_h.param_groups:
        #     for param in group['params']:
        #         print(id(param))
        # print("----------------------------------------------------")
        # print(self.cluster_model.degradation.H)


    # 3个训练步骤中，值参数值
    def configure_optimizers(self) -> dict:
        optims = []
        ae_params = []
        for iter in self.cluster_model.encoder_decoder:
            ae_params.extend(iter.parameters())
        # 退化网络除了H的所有参数
        dg_params = [p for name, p in self.cluster_model.degradation.named_parameters() if 'H' not in name]
       
        optims.append(optim.Adam(ae_params, lr = self.lr_ae))
        optims.append(optim.Adam(dg_params, lr = self.lr_dg))

        # H的引用改变了，优化器中的H和新的H不指向同一块地方，优化器无法更改H
        # optims.append(optim.Adam(self.cluster_model.degradation.parameters(), lr = self.lr_h))
        return optims


if __name__ == '__main__':
    pass