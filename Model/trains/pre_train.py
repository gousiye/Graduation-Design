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

class PreTrain(pl.LightningModule):

    def __init__(
        self, 
        encoder_decoder: List[EncoderDecoder],
        lr: float,
        cluster_num: int,
        description: dict
    ) -> None:
        
        super(PreTrain, self).__init__()
        self.encoder_decoder = encoder_decoder
        self.lr = lr
        self.cluster_num = cluster_num
        self.description = description
        self.loss = [0]* len(self.encoder_decoder)
        self.min_loss = float('inf')
        self.best_preModel_state = [None] * len(self.encoder_decoder)
        self.best_epoch_idx = -1
        self.config = {}

    def __SaveBetterModel(self):
        for i in range(len(self.encoder_decoder)):
            self.best_preModel_state[i] = self.encoder_decoder[i].state_dict()

    def on_train_start(self) -> None:
        print()
        print("----------------------预训练开始---------------------")
        print()
        

    # pytorch_lightning 只要在training_step中正确计算，优化器参数列表包含就可以。这样也是可以的
    def forward(self, features:List[Tensor]):
        ae_pre_loss_tmp = 0
        for i in range(len(features)):
            tmpMSE = nn.MSELoss()(features[i], self.encoder_decoder[i](features[i]))
            self.loss[i] += tmpMSE
            ae_pre_loss_tmp += tmpMSE
        return ae_pre_loss_tmp

    def training_step(self, batch:Tuple, batch_idx:int):
        features, _ = batch
        ae_pre_loss = self.forward(features)
        self.log('ae_pre_loss', ae_pre_loss)
        return {'loss':ae_pre_loss}
    
    def on_train_epoch_start(self) -> None:
        for i in range(len(self.loss)):
            self.loss[i] = 0

    def training_epoch_end(self, outputs: List) -> None:
        current_epoch = self.current_epoch
        # tensor.mean 默认会计算全局平均
        ae_pre_loss = torch.stack([x['loss'] for x in outputs]).mean()
        print()
        output = "Pre_epoch: {:.0f}, Mean: ae_pre_loss: {:.4f}.  Sum: ".format(current_epoch, ae_pre_loss)
        for i in range(len(self.loss)):
            if i != len(self.loss) - 1:
                output += "loss{:.0f} = {:.4f}, ".format(i , self.loss[i])
            else:
                 output += "loss{:.0f} = {:.4f}.".format(i , self.loss[i])
        print(output)

        # 保存最优的模型
        if self.best_epoch_idx == -1 or ae_pre_loss < self.min_loss:
            self.best_epoch_idx = current_epoch
            self.min_loss = ae_pre_loss
            self.__SaveBetterModel()
    
    def on_train_end(self):
        print()
        print("最好的预训练模型在第{i}个epoch, 其ae_pre_loss为{loss:.4f}".format(i = self.best_epoch_idx, loss = self.min_loss) )
        print("----------------------预训练结束---------------------")
        print()
        for i in range(len(self.encoder_decoder)):
            self.encoder_decoder[i].load_state_dict(self.best_preModel_state[i])
            pass

    def configure_optimizers(self):
        params = []
        for iter in self.encoder_decoder:
            params.extend(iter.parameters())
        return optim.Adam(params, lr = self.lr)

    
    def Save(self, config:dict) -> None:
        """
        保存模型,日志
        """
        model_path = config['model_params']['pre_model_path']
        log_path = config['log_params']['pre_log_path']
        data_name = config['data_params']['data_name']

        description = {}

        now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        check_folder_path = os.path.join(model_path, data_name)    
        check_model_path = os.path.join(check_folder_path, data_name) + '.pth.tar'
        check_param_path = os.path.join(check_folder_path, data_name) + '.yaml'

        log_folder_path = os.path.join(log_path, data_name + now_time)    
        log_model_path = os.path.join(log_folder_path, data_name) + '.pth.tar'
        log_param_path = os.path.join(log_folder_path, data_name) + '.yaml'

        hyper_parameters = config
        model_parameters = ParameterTool.GetPreModelDescription(self.encoder_decoder)

        if not os.path.exists(check_folder_path):
            os.makedirs(check_folder_path)
        if not os.path.exists(log_folder_path):
            os.makedirs(log_folder_path)
        loss_parameters = ParameterTool.GetPreLossDescription(self.min_loss, self.loss)        

        FileTool.SavePreModel(check_model_path, self.encoder_decoder, loss_parameters)
        FileTool.SavePreModel(log_model_path, self.encoder_decoder, loss_parameters)

        description.update(hyper_parameters)
        description.update(loss_parameters)
        description.update(model_parameters)

        FileTool.SaveConfigYaml(check_param_path, description)
        FileTool.SaveConfigYaml(log_param_path, description)
        
        print("------------------------")
        print('\033[94m' + "已保存预训练AE模型到{path}".format(path = check_model_path) + '\033[0m')
        print("------------------------")

       
       
    def Load(self, config):
        """
        读取模型，日志
        """
        model_path = config['model_params']['pre_model_path']
        data_name = config['data_params']['data_name']

        description = {}

        check_folder_path = os.path.join(model_path, data_name)    
        check_model_path = os.path.join(check_folder_path, data_name) + '.pth.tar'
        check_param_path = os.path.join(check_folder_path, data_name) + '.yaml'

        assert os.path.exists(check_model_path), '该模型没有存储文件'
        loss_parameters = {}
        FileTool.LoadPreModel(check_model_path, self.encoder_decoder, loss_parameters)
        print("------------------------")
        print('\033[94m' + "已从{path}读取预训练AE模型".format(path = check_model_path) + '\033[0m')
        print("------------------------")
        
        hyper_parameters = config
        model_parameters = ParameterTool.GetPreModelDescription(self.encoder_decoder)

        description.update(hyper_parameters)
        description.update(loss_parameters)
        description.update(model_parameters)
        FileTool.SaveConfigYaml(
            check_param_path,
            description
        )

if __name__ == '__main__':
    pass