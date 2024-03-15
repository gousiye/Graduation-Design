from trains import PreTrain
from trains import FirstTrain
from trains import SecondTrain
from models import ClusterModel
from pytorch_lightning import Trainer
from utils import ClusterDataset, FileTool, ParameterTool, Metric
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import torch
import warnings
import random
import os
import numpy as np
from torch import optim
import torch.nn as nn
import itertools
from trains import test

# 忽略特定类型的警告
warnings.filterwarnings("ignore", category=UserWarning, message=".*reduction.*")
warnings.filterwarnings("ignore", category=UserWarning, message="The dataloader.*does not have many workers")
warnings.filterwarnings("ignore", category=UserWarning, message=".*LightningModule.configure_optimizers.*")

random_seed = 100
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(random_seed)


class Train():
    def __init__(
        self,
        cluster_model:ClusterModel,
        cluster_dataset: ClusterDataset,
        config: dict
    ):
        self.cluster_model = cluster_model
        self.cluster_dataset = cluster_dataset
        self.config = config
        self.pre_train = None
        self.first_train = None
        self.second_train = None
        self.cluster_model.GenerateH(self.cluster_dataset.GetLen(), self.config['model_params']['H_dim'])
        ParameterTool.InitVarFromDict(self, config)
        self.__toCuda()
        self.dataloader = DataLoader(
            dataset = self.cluster_dataset.dataset,
            batch_size = 400,
            num_workers = 0
        )


    def __toCuda(self):
        for iter in self.cluster_model.encoder_decoder:
            iter.encoders.to(f'cuda:{self.devices[0]}')
            iter.decoders.to(f'cuda:{self.devices[0]}')
            # iter.center = iter.center.to(f'cuda:{self.devices[0]}')
            pass
        pass

    def __PreTrain(self, encoder_decoder):
        """
        训练或者读取, 得到预训练的模型, 初始化AE的参数
        """
        # test.TestTrain(encoder_decoder, self.dataloader)

        self.pre_train = PreTrain(
                self.cluster_model.encoder_decoder,
                self.pre_lr_ae,
                self.cluster_num,
                self.config,
        )
        pre_config = ParameterTool.PreDescription(self.config)
    
        if self.is_pre_train == True:
            preTrainer = Trainer(
                logger = False,
                callbacks = [ModelCheckpoint(save_last=False, save_top_k=0, monitor=None)],
                accelerator = self.accelerator,
                devices = self.devices,
                max_epochs = self.pre_max_epochs)
            preTrainer.fit(self.pre_train, self.cluster_dataset)

            if self.save_pre_model:
                pass
                self.pre_train.Save(pre_config)
            else:
                pass
        
        else:
            self.pre_train.Load(pre_config)




    def __FirstTrain(self):
        """
        训练或者读取, 得到第一次训练的模型, 训练AE, DG, H 
        """
        self.first_train = FirstTrain(self.cluster_model, self.cluster_dataset, self.config)

        if self.is_first_train == True:
            firstTrainer = Trainer(
                logger = False,
                callbacks = [ModelCheckpoint(save_last=False, save_top_k=0, monitor=None)],
                accelerator = self.accelerator,
                devices = self.devices,
                max_epochs = self.first_total_max_epochs)
            firstTrainer.fit(self.first_train, self.cluster_dataset)

            if self.save_first_model:
                self.first_train.Save()
                pass
            else:
                pass
        
        else:
            self.first_train.Load()

        # print(self.cluster_dataset.dataset.H)
            # self.pre_train.Load(pre_config)
    
    def __SecondTrain(self):
        self.second_train = SecondTrain(self.cluster_model, self.cluster_dataset, self.config)
        
        if self.is_second_train == True:
            secondTrainer = Trainer(
            logger = False,
            callbacks = [ModelCheckpoint(save_last=False, save_top_k=0, monitor=None)],
            accelerator = self.accelerator,
            devices = self.devices,
            max_epochs = self.second_total_max_epochs)
            secondTrainer.fit(self.second_train, self.cluster_dataset)
            
            if self.save_second_model:
                pass
            else:
                pass
        else:
            pass
     
    def StartTrain(self):
        if self.is_first_train == True:
            self.__PreTrain(self.cluster_model.encoder_decoder)
        self.__FirstTrain()
        self.__SecondTrain()

    