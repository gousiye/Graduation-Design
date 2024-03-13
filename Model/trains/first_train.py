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
from trains import PreTrain
from pytorch_lightning import Trainer
import os
from pytorch_lightning.callbacks import ModelCheckpoint
import datetime

class FirstTrain(pl.LightningModule):
    def __init__(
        self, 
        cluster_model:ClusterModel,
        cluster_dataset: ClusterDataset,
        config: dict
    )->None:
        super(FirstTrain, self).__init__()
        self.automatic_optimization = False
        self.cluster_model = cluster_model
        self.cluster_dataset = cluster_dataset
        self.config = config

        self.min_h_loss = float('inf')
        self.ae_loss = float('inf')
        self.dg_loss  = float('inf')

        self.best_degradation_state = None 
        # self.cluster_model.state_dict()只有degradation相关的参数，encoder_decoder需要手动维护
        self.best_AeModel_state = [None] * len(self.cluster_model.encoder_decoder)
        self.best_epoch_idx = -1
        ParameterTool.InitVarFromDict(self, self.config)

    def forward(self):
        return self.cluster_model
    
    def on_train_start(self) -> None:
        print()
        print("----------------------训练第一步开始------------------------")
        print()
    
    # 保存更好的模型
    def __SaveBetterState(self):
        self.best_degradation_state = self.cluster_model.degradation.state_dict()
        for i in range(len(self.cluster_model.encoder_decoder)):
            self.best_AeModel_state[i] = self.cluster_model.encoder_decoder[i].state_dict()

    # 更新cluster_dataset.dataset上的H，下一轮的epoch要从这读取
    def __update_H(self, batch_idx:int) -> None:
        new_batch_h = None
        if self.cluster_model.degradation.h.is_cuda:
            new_batch_h = self.cluster_model.degradation.h.cpu()
        else:
            new_batch_h = self.cluster_model.degradation.h

        # batch_idx从0开始的
        start_idx = batch_idx * self.batch_size   
        end_idx = start_idx + new_batch_h.shape[0]
        
        self.cluster_model.H[start_idx: end_idx, ...] = new_batch_h.detach()


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
        self.cluster_model.degradation.h.requires_grad_(False)
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
        self.cluster_model.degradation.h.requires_grad_(True)
        opt_h = (optim.Adam(self.cluster_model.degradation.parameters(), lr = self.lr_h))
        h_loss_avg = 0
        # 单独训练H
        for i in range(self.first_h_max_epochs):
            opt_h.zero_grad()        
            # dg, ae网络的参数有变化，因此需要重新计算
            view_h_list = self.cluster_model.degradation.get_view_h_list()
            z_half_list =  self.cluster_model.get_z_half_list(features)
            h_loss = Metric.GetAverageMSE(z_half_list, view_h_list)
            h_loss_avg += h_loss
            self.manual_backward(h_loss)
            opt_h.step()
        h_loss_avg /= self.first_h_max_epochs
        return h_loss_avg 


    def training_step(self, batch: Tuple, batch_idx):
        features, labels = batch
        start_idx = batch_idx * self.batch_size   
        end_idx = start_idx + labels.shape[0]
        h = self.cluster_model.H[start_idx:end_idx, ...].to(f'cuda:{self.devices[0]}')
        self.cluster_model.degradation.set_h(h)
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
        
        current_epoch = self.current_epoch
        ae_mean_loss = torch.stack([x['ae_loss'] for x in outputs]).mean()
        dg_mean_loss = torch.stack([x['dg_loss'] for x in outputs]).mean()
        h_mean_loss = torch.stack([x['h_loss'] for x in outputs]).mean()
        output = "First_epoch: {:.0f}, ae_loss_mean: {:.4f}, dg_mean_loss:{:.4f}, h_mean_loss:{:.4f}.". \
            format(current_epoch, ae_mean_loss, dg_mean_loss, h_mean_loss)
        print(output)

        # 保存最h_mean_loss最小的模型
        if h_mean_loss < self.min_h_loss:
            self.best_epoch_idx = current_epoch
            self.min_h_loss = h_mean_loss
            self.ae_loss = ae_mean_loss
            self.dg_loss = dg_mean_loss
            self.__SaveBetterState()
            
    def on_train_end(self):
        print()
        print("最好的第一步训练模型在第{i}个epoch, 其h_loss_mean为{loss:.4f}".format(i = self.best_epoch_idx, loss = self.min_h_loss))
        print("----------------------训练第一步结束---------------------")
        print()
        self.cluster_model.degradation.load_state_dict(self.best_degradation_state)
        # degradateion.h已经没有用了，这样只是便于存储和读取
        self.cluster_model.degradation.h = torch.nn.Parameter(torch.FloatTensor(self.batch_size, self.H_dim))
        for i in range(len(self.cluster_model.encoder_decoder)):
            self.cluster_model.encoder_decoder[i].load_state_dict(self.best_AeModel_state[i])


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


    def Save(self):
        """
        存储模型，日志
        """
        path = {}
        loss = {}
        path['model_path'] = self.first_model_path
        path['log_path'] = self.first_log_path
        path['data_name'] = self.data_name
        loss['ae_loss'] = self.ae_loss.item()
        loss['dg_loss'] = self.dg_loss.item()
        loss['h_loss'] = self.min_h_loss.item()
        description = ParameterTool.Description(self.config, 'first')
        FileTool.SaveModelAndLog(self.cluster_model, description, path, loss)

        check_folder_path = os.path.join(self.first_model_path, self.data_name)    
        check_model_path = os.path.join(check_folder_path, self.data_name) + '.pth.tar'
        print("------------------------")
        print('\033[94m' + "已保存预训练AE模型到{path}".format(path = check_model_path) + '\033[0m')
        print("------------------------")

    def Load(self):
        """
        读取模型, 生成相关日志
        """
        path = {}
        path['model_path'] = self.first_model_path
        path['log_path'] = self.first_log_path
        path['data_name'] = self.data_name
        description = ParameterTool.Description(self.config, 'first')
        FileTool.LoadAndLog(self.cluster_model, description, path)

        check_folder_path = os.path.join(self.first_model_path, self.data_name)    
        check_model_path = os.path.join(check_folder_path, self.data_name) + '.pth.tar'
        print("------------------------")
        print('\033[94m' + "已从{path}读取第一次训练模型".format(path = check_model_path) + '\033[0m')
        print("------------------------")
        

if __name__ == '__main__':
    pass