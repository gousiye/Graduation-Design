import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
import torch
from utils import ClusterDataset, FileTool, ParameterTool
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
        self.cluster_model = cluster_model
        self.cluster_dataset = cluster_dataset
        self.config = config
        self.InitVariables()
        self.GetPreModel()


    def InitVariables(self):
        """
        把config.yaml中的参数转为类中的成员
        """
        for param_aspect in self.config:
            for parameter in self.config[param_aspect]:
                setattr(self, parameter, self.config[param_aspect][parameter])


    def GetPreModel(self):
        """
        训练或者读取, 得到预训练的模型, 初始化AE的参数
        """
        description = {'a' :12}
        now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        pre_folder_path = os.path.join(self.pre_model_path, self.data_name)    
        pre_model_path = os.path.join(pre_folder_path, self.data_name) + '.pth.tar'
        pre_param_path = os.path.join(pre_folder_path, self.data_name) + '.yaml'

        if not os.path.exists(pre_folder_path):
            os.makedirs(pre_folder_path)
        if self.is_pre_train == True:  
            pretrain = PreTrain(
                self.cluster_model.encoder_decoder,
                self.lr_pre,
                self.cluster_num,
                self.config
            )

            preTrainer = Trainer(
                logger = False,
                callbacks=[ModelCheckpoint(save_last=False, save_top_k=0, monitor=None)],
                **self.config['pre_trainer_params'])
            preTrainer.fit(pretrain, self.cluster_dataset)
            
            hyper_parameters = ParameterTool.CutDescription(self.config)
            model_parameters = ParameterTool.GetModelDescription(self.cluster_model, is_pre = True)
            loss_parameters = {}
            description = {}

            # 保存训练的模型
            if self.save_pre_model == True:
                loss_parameters = ParameterTool.GetPreLossDescription(pretrain.min_loss, pretrain.loss)
                
                log_folder_path = os.path.join(self.pre_log_path, self.data_name)    + now_time 
                log_model_path = os.path.join(log_folder_path, self.data_name) + '.pth.tar'
                log_parm_path = os.path.join(log_folder_path, self.data_name) + '.yaml'
                
                if not os.path.exists(log_folder_path):
                    os.makedirs(log_folder_path)

                FileTool.SaveModel(pre_model_path, self.cluster_model.encoder_decoder, loss_parameters)
                FileTool.SaveModel(log_model_path, self.cluster_model.encoder_decoder, loss_parameters)

                description.update(hyper_parameters)
                description.update(loss_parameters)
                description.update(model_parameters)

                FileTool.SaveConfigYaml(pre_param_path,description)
                FileTool.SaveConfigYaml(log_parm_path,description)
                
                print("------------------------")
                print('\033[94m' + "已保存预训练AE模型到{path}".format(path = pre_model_path) + '\033[0m')
                print("------------------------")

        else:   
            assert os.path.exists(pre_model_path), '该模型没有存储文件'
            FileTool.LoadModel(pre_model_path, self.cluster_model, loss_parameters)
            print("------------------------")
            print('\033[94m' + "已从{path}读取预训练AE模型".format(path = pre_model_path) + '\033[0m')
            print("------------------------")
            
            description.update(hyper_parameters)
            description.update(loss_parameters)
            description.update(model_parameters)
            FileTool.SaveConfigYaml(
                pre_param_path,
                description
            )

if __name__ == '__main__':
    pass