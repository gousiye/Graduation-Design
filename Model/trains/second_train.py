import pytorch_lightning as pl
import torch
from utils import ClusterDataset, FileTool, ParameterTool, Metric
from typing import Tuple
import torch.nn as nn
from models import ClusterModel
from typing import List, Optional, Sequence, Union, Any, Callable, Dict
from torch import Tensor
from torch import optim
import os
from pytorch_lightning.callbacks import ModelCheckpoint
import datetime
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans

class SecondTrain(pl.LightningModule):
    def __init__(
        self, 
        cluster_model:ClusterModel,
        cluster_dataset: ClusterDataset,
        config: dict
    )->None:
        super(SecondTrain, self).__init__()
        self.automatic_optimization = False
        self.config = config
        ParameterTool.InitVarFromDict(self, self.config)
        self.cluster_model = cluster_model
        self.cluster_dataset = cluster_dataset
        self.random_seed = 100  # kMeans的随机种子，设置成固定的，有一定稳定性
        self.loss_parameters = {} 
        self.model_parameters = {} 
        self.dataloader = DataLoader(
            dataset = self.cluster_dataset.dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers
        )

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

    def __get_center(self, view_num) -> Tuple:
        """
        获得每个视角的聚类质心, 公共表示H的聚类质心
        """
        batch_features = [[] for _ in range(view_num)]
        final_features = [None] * view_num
        centers = [None] * view_num
        centerH = None
        for features, labels in self.dataloader:
            labels = labels.to(f'cuda:{self.devices[0]}')
            for i in range(view_num):   
                batch_features[i].append(self.cluster_model.encoder_decoder[i].get_z_half(features[i].to(f'cuda:{self.devices[0]}')))
        
        final_H = self.cluster_model.H
        
        for i in range(view_num):
            final_features[i] = torch.cat(batch_features[i], 0)

        for i in range(view_num):
            km = KMeans(n_clusters=self.cluster_num, random_state=self.random_seed,n_init = 10)
            km.fit_predict(final_features[i].detach().cpu().numpy())
            centers[i] = torch.from_numpy(km.cluster_centers_).to(f'cuda:{self.devices[0]}')

        km = KMeans(n_clusters=self.cluster_num, random_state=self.random_seed,n_init = 10)
        km.fit_predict(final_H.detach().cpu().numpy())
        centerH = torch.from_numpy(km.cluster_centers_).to(f'cuda:{self.devices[0]}')
        return centers, centerH
    
    def on_train_start(self) -> None:
        print()
        print("----------------------训练第二步开始------------------------")
        print()

    def on_train_epoch_start(self):
        centers, centerH = self.__get_center(self.view_num)
        for i in range(len(self.cluster_model.encoder_decoder)):
            self.cluster_model.encoder_decoder[i].set_center(centers[i])
        self.cluster_model.degradation.set_center(centerH)

    def train_ae(self, features, h):
        ae_params = []
        for iter in self.cluster_model.encoder_decoder:
            ae_params.extend(iter.parameters())
        opt_ae = optim.Adam(ae_params, lr = self.second_lr_ae)
        opt_ae.zero_grad()

        q_h = self.cluster_model.degradation.get_q(h)
        p_h = self.cluster_model.degradation.get_p(q_h) 
        
        z_half_list = self.cluster_model.get_z_half_list(features)
        view_h_list = self.cluster_model.degradation.get_view_h_list()
        ae_recon_list = self.cluster_model.get_ae_recon_list(features)
        q_ae_list = self.cluster_model.get_ae_q_list(z_half_list)

        ae_recon_loss = Metric.GetAverageMSE(features, ae_recon_list)
        ae_degrade_loss = Metric.GetAverageMSE(view_h_list, z_half_list)
        tmp_dec_loss =  0
        for i in range(len(q_ae_list)):
            tmp_dec_loss += nn.KLDivLoss()(p_h.log(), q_ae_list[i]) # 第一个参数是目标分布的对数分布
        
        dec_loss = tmp_dec_loss
        ae_loss = ae_recon_loss + ae_degrade_loss + dec_loss
        ae_loss.backward()
        opt_ae.step()
        return ae_loss


    def train_h(self, features, h):
        """
        训练dg和h
        """
        for v in self.cluster_model.degradation.parameters():
            v.requires_grad = True

        opt_h = optim.Adam(params=self.cluster_model.degradation.parameters(), lr=self.second_lr_h)
        h_loss_total = 0
        for k in range(self.second_h_max_epochs):
            opt_h.zero_grad()
            q_h = self.cluster_model.degradation.get_q(h)
            p_h = self.cluster_model.degradation.get_p(q_h) 
    
            z_half_list  = self.cluster_model.get_z_half_list(features)
            view_h_list = self.cluster_model.degradation.get_view_h_list()
            h_recon_loss = Metric.GetAverageMSE(z_half_list, view_h_list)
            h_dec_loss = nn.KLDivLoss()(p_h.log(), q_h)
            h_loss = h_recon_loss + h_dec_loss
            h_loss_total += h_loss
            h_loss.backward()
            opt_h.step()
        return h_loss_total

    def training_step(self, batch: Tuple, batch_idx):
        features, labels = batch
        start_idx = batch_idx * self.batch_size   
        end_idx = start_idx + labels.shape[0]
        h = self.cluster_model.H[start_idx:end_idx, ...].to(f'cuda:{self.devices[0]}')
        self.cluster_model.degradation.set_h(h)
        ae_loss = self.train_ae(features, h)
        h_loss = self.train_h(features, h)
        self.log('ae_loss', ae_loss)
        self.log('h_loss', h_loss)
        self.__update_H(batch_idx)
        return {'loss':h_loss, 'ae_loss':ae_loss, 'h_loss':h_loss}


    def training_epoch_end(self, outputs: List) -> None:
        print()
        current_epoch = self.current_epoch
        ae_loss = torch.stack([x['ae_loss'] for x in outputs]).sum()
        h_loss = torch.stack([x['h_loss'] for x in outputs]).sum()
        self.ae_loss = ae_loss
        self.h_loss = h_loss
        output = "Second_epoch: {:.0f}, ae_loss: {:.4f}, h_loss:{:.4f}.". \
            format(current_epoch, ae_loss, h_loss)
        print(output)
        # 为了读取的时候时候维度一致，实际的训练中会先设置h，然后再训练
        self.cluster_model.degradation.set_h(torch.FloatTensor(self.batch_size, self.H_dim))

    def on_train_end(self):
        self.cluster_model.degradation.h = torch.nn.Parameter(torch.FloatTensor(self.batch_size, self.H_dim))
        loss_parameters = {}
        loss_parameters['ae_loss'] = self.ae_loss.item()
        loss_parameters['h_loss'] = self.h_loss.item()
        self.loss_parameters['loss'] = loss_parameters
        self.model_parameters = ParameterTool.GetModelDescription(self.cluster_model)
        print()
        print("----------------------训练第二步结束---------------------")
        print()


    # ae中的center， dg中的h 每次都会改变，每个epoch都需要重新设置优化器
    def configure_optimizers(self) -> dict:
        return None

    def UpdateYaml(self,yaml):
        """
        更新yaml描述
        """
        yaml['second_train'] = {}
        field = {
            'second_trainer_params':['second_lr_ae', 'second_lr_dg', 'second_lr_h', 'second_total_max_epochs','second_h_max_epochs']
        }
        params = ParameterTool.GetDescription(self.config, field)
        yaml['second_train'].update(params)
        yaml['second_train'].update(self.loss_parameters)
        yaml['second_train'].update(self.model_parameters)
        # FileTool.SaveConfigYaml("test.yaml", yaml)     
    
    def SaveModel(self):
        """
        存储模型，
        """
        FileTool.SaveModel(os.path.join(self.model_path,'second_train')+'.pth.tar', self.cluster_model, self.loss_parameters)
        if self.is_second_train == True:
            print("------------------------")
            print('\033[94m' + "已保存第二次训练模型到{path}".format(path = self.model_path ) + '\033[0m')
            print("------------------------")

    def SaveLog(self):
        """
        保存日志
        """
        FileTool.SaveModel(os.path.join(self.log_path,'second_train')+'.pth.tar', self.cluster_model, self.loss_parameters)

    def Load(self):
        """
        读取模型, 生成相关日志
        """
        self.loss_parameters = {}
        self.model_parameters = ParameterTool.GetModelDescription(self.cluster_model)
        model_file = os.path.join(self.model_path,'second_train') + '.pth.tar'
        assert os.path.exists(model_file), "模型文件不存在"
        FileTool.LoadModel(model_file, self.cluster_model, self.loss_parameters)
        print("------------------------")
        print('\033[94m' + "已从{path}读取第二次训练模型".format(path = self.model_path) + '\033[0m')
        print("------------------------")


