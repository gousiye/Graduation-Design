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
        self.__GetPreModel()


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
                callbacks=[ModelCheckpoint(save_last=False, save_top_k=0, monitor=None)],
                **self.config['pre_trainer_params'])
            preTrainer.fit(pretrain, self.cluster_dataset)

        pretrain.ProcessEnd(
            ParameterTool.PreDescription(self.config), 
            self.is_pre_train,
            self.save_pre_model
        )

    def forward(self):
        return self.cluster_model
    
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
        for para_name, para_tensor in self.cluster_model.degradation.named_parameters():
            if para_name == 'H':
               para_tensor.requires_grad = False
        opt_dg = opt_ae = self.optimizers()[1]

    def training_step(self, batch: Tuple, batch_size):
        features, labels, h = batch
        self.cluster_model.degradation.set_H(h)
        ae_loss = self.train_ae(features)
        self.train_dg(features)
        self.log('ae_loss', ae_loss)
        # print()
        # print(ae_loss)
        return {'loss':ae_loss, 'ae_loss':ae_loss}

    def training_epoch_end(self, outputs: List) -> None:
        print()
        for para in self.cluster_model.degradation.parameters():
            print(para.shape)
        # opt_dg = self.optimizers()[1]
        
        # for param_group in opt_dg.param_groups:
        #     for param in param_group['params']:
        #         print(param.name, param.shape, param.dtype)
        # pass

    

    def configure_optimizers(self) -> dict:
        optims = []
        ae_params = []
        for iter in self.cluster_model.encoder_decoder:
            ae_params.extend(iter.parameters())
        optims.append(optim.Adam(ae_params, lr = self.lr_ae))
        optims.append(optim.Adam(ae_params, lr = self.lr_ae))
        # optims.append(optim.Adam(ae_params, lr = self.lr_ae))
        # optims.append(optim.Adam(self.cluster_model.degradation.parameters(), lr = self.lr_ae))
        return optims


if __name__ == '__main__':
    pass