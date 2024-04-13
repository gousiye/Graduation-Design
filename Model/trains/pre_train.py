import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from typing import Tuple
import torch.nn as nn
from models.submodels import EncoderDecoder
from typing import List, Optional, Sequence, Union, Any, Callable, Dict
import torch
from torch import Tensor
from torch import optim
import datetime
import os
from utils import ClusterDataset, FileTool, ParameterTool
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from utils import ClusterDataset, FileTool, ParameterTool, Metric

class PreTrain(pl.LightningModule):

    def __init__(
        self, 
        encoder_decoder: List[EncoderDecoder],
        lr: float,
        cluster_num: int,
        config: dict
    ) -> None:
        
        super(PreTrain, self).__init__()

        self.encoder_decoder = encoder_decoder
        self.lr = lr
        self.cluster_num = cluster_num
        self.config = config
        ParameterTool.InitVarFromDict(self, self.config) 
        self.loss = [0]* len(self.encoder_decoder)
        self.min_loss = float('inf')
        self.best_preModel_state = [None] * len(self.encoder_decoder)
        self.best_epoch_idx = -1
        self.model_parameters = {}
        self.loss_parameters = {}

    def __SaveBetterModel(self):
        for i in range(len(self.encoder_decoder)):
            self.best_preModel_state[i] = self.encoder_decoder[i].state_dict()

    def on_train_start(self) -> None:
        print()
        print("----------------------预训练开始---------------------")
        print()
        

    # pytorch_lightning 只要在training_step中正确计算，优化器参数列表包含就可以。这样也是可以的
    def forward(self, features:List[Tensor]):
        pass


    def training_step(self, batch:Tuple, batch_idx:int):
        ae_params = []
        for iter in self.encoder_decoder:
            ae_params.extend(iter.parameters())
        opt_ae = optim.Adam(ae_params, lr = self.lr)
        features, _ = batch
        ae_pre_loss_tmp = 0
        for i in range(len(features)):
            tmpMSE = nn.MSELoss()(features[i], self.encoder_decoder[i](features[i]))
            self.loss[i] += tmpMSE
            ae_pre_loss_tmp += tmpMSE
        ae_pre_loss = ae_pre_loss_tmp
        self.log('ae_pre_loss', ae_pre_loss)
        return {'loss':ae_pre_loss}
    
    def on_train_epoch_start(self) -> None:
        for i in range(len(self.loss)):
            self.loss[i] = 0

    def training_epoch_end(self, outputs: List) -> None:
        current_epoch = self.current_epoch
        # tensor.sum 默认会计算全局和
        ae_pre_loss = torch.stack([x['loss'] for x in outputs]).sum()
        print()
        output = "Pre_epoch: {:.0f}, ae_pre_loss: {:.4f}: ".format(current_epoch, ae_pre_loss)
        for i in range(len(self.loss)):
                output += ", loss{:.0f} = {:.4f}".format(i , self.loss[i])
        print(output)
        

        # 保存最优的模型
        if self.best_epoch_idx == -1 or ae_pre_loss < self.min_loss:
            self.best_epoch_idx = current_epoch
            self.min_loss = ae_pre_loss
            self.__SaveBetterModel()
    
    def on_train_end(self):
        self.model_parameters = ParameterTool.GetPreModelDescription(self.encoder_decoder)
        self.loss_parameters = ParameterTool.GetPreLossDescription(self.min_loss, self.loss) 
        for i in range(len(self.encoder_decoder)):
            self.encoder_decoder[i].load_state_dict(self.best_preModel_state[i])
        print()
        print("最好的预训练模型在第{i}个epoch, 其ae_pre_loss为{loss:.4f}".format(i = self.best_epoch_idx, loss = self.min_loss) )
        print("----------------------预训练结束---------------------")
        print()

    def configure_optimizers(self):
        ae_params = []
        for iter in self.encoder_decoder:
            ae_params.extend(iter.parameters())
        opt_ae_pre = optim.Adam(params=ae_params, lr = self.lr)
        return opt_ae_pre
    
    def UpdateYaml(self, yaml):
        """
        更新.yaml描述
        """
        yaml['pre_train'] = {}
        field = {
            'pre_trainer_params':['pre_lr_ae', 'pre_max_epochs']
        }
        params = ParameterTool.GetDescription(self.config, field)
        yaml['pre_train'].update(params)
        yaml['pre_train'].update(self.loss_parameters)
        yaml['pre_train'].update(self.model_parameters)
        # FileTool.SaveConfigYaml("test.yaml", yaml)

    def SaveMoel(self) -> None:
        """
        保存模型
        """
        FileTool.SavePreModel(os.path.join(self.model_path,'pre_train') + '.pth.tar', self.encoder_decoder, self.loss_parameters)
        if self.is_pre_train == True:
            print("------------------------")
            print('\033[94m' + "已保存预训练AE模型到{path}".format(path = self.model_path) + '\033[0m')
            print("------------------------")

    def SaveLog(self)->None:
        """
        保存日志
        """
        FileTool.SavePreModel(os.path.join(self.log_path,'pre_train') + '.pth.tar', self.encoder_decoder, self.loss_parameters)
        
    def Load(self):
        """
        读取模型，日志
        """
        self.loss_parameters = {}
        self.model_parameters = ParameterTool.GetPreModelDescription(self.encoder_decoder)
        model_file = os.path.join(self.model_path, 'pre_train') + '.pth.tar'
        assert os.path.exists(model_file), '该模型没有存储文件'
        FileTool.LoadPreModel(model_file, self.encoder_decoder, self.loss_parameters)
        print("------------------------")
        print('\033[94m' + "已从{path}读取预训练AE模型".format(path = self.model_path) + '\033[0m')
        print("------------------------")