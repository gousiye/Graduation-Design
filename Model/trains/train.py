import scipy.io
from trains import PreTrain
from trains import FirstTrain
from trains import SecondTrain
from models import ClusterModel
from pytorch_lightning import Trainer
from utils import ClusterDataset,  ParameterTool, Metric, Cluster, FileTool
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import warnings
import random
import os
import numpy as np
import scipy
import h5py

# 忽略特定类型的警告
warnings.filterwarnings("ignore", category=UserWarning, message=".*reduction.*")
warnings.filterwarnings("ignore", category=UserWarning, message="The dataloader.*does not have many workers")
warnings.filterwarnings("ignore", category=UserWarning, message=".*LightningModule.configure_optimizers.*")

# 固定随机种子
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
        self.yaml = None
        self.pre_train = None
        self.first_train = None
        self.second_train = None
        self.y_predcit_soft = None
        self.y_true = self.cluster_dataset.dataset.y
        self.soft_assign_loss = {} # 软分配的各指标
        self.cluster_loss_avg = {} # 模拟聚类的各指标的平均值
        self.cluster_model.GenerateH(self.cluster_dataset.GetLen(), self.config['model_params']['H_dim'])
        ParameterTool.InitVarFromDict(self, config)
        self.__toCuda()
        
    def __toCuda(self):
        for iter in self.cluster_model.encoder_decoder:
            iter.encoders.to(f'cuda:{self.devices[0]}')
            iter.decoders.to(f'cuda:{self.devices[0]}')

    def __SetFolder(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def __InitialYaml(self):
        field = {
            'model_params': ['latent_encoder_dim','H_dim'],
            'data_params':['data_name', 'view_num', 'cluster_num', 'batch_size', 'num_workers'],
            'device_params':['accelerator', 'devices'],
        }
        self.yaml = ParameterTool.GetDescription(self.config, field)

    def __PreTrain(self):
        """
        训练或者读取, 得到预训练的模型, 初始化AE的参数
        """
        self.pre_train = PreTrain(
                self.cluster_model.encoder_decoder,
                self.pre_lr_ae,
                self.cluster_num,
                self.config,
        )
        if self.is_pre_train == True:
            preTrainer = Trainer(
                logger = False,
                callbacks = [ModelCheckpoint(save_last=False, save_top_k=0, monitor=None)],
                accelerator = self.accelerator,
                devices = self.devices,
                max_epochs = self.pre_max_epochs)
            preTrainer.fit(self.pre_train, self.cluster_dataset)        
        else:
            self.pre_train.Load()

        # 重新保存，为了文件的修改时间一致
        if self.save_model:
            self.pre_train.SaveMoel()

        if self.save_log:
            self.pre_train.SaveLog()
        self.pre_train.UpdateYaml(self.yaml)


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
        else:
            self.first_train.Load()
        if self.save_model:
            self.first_train.SaveModel()
        if self.save_log:
            self.first_train.SaveLog()
        self.first_train.UpdateYaml(self.yaml)
        

    def __SecondTrain(self):
        """
        训练或者读取, 得到第二次训练的模型, 训练AE, DG, H 
        """
        self.second_train = SecondTrain(self.cluster_model, self.cluster_dataset, self.config)
        
        if self.is_second_train == True:
            secondTrainer = Trainer(
            logger = False,
            callbacks = [ModelCheckpoint(save_last=False, save_top_k=0, monitor=None)],
            accelerator = self.accelerator,
            devices = self.devices,
            max_epochs = self.second_total_max_epochs)
            secondTrainer.fit(self.second_train, self.cluster_dataset)
        else:
            self.second_train.Load()
        if self.save_model:
            self.second_train.SaveModel()
        if self.save_log:
            self.second_train.SaveLog() 
        self.second_train.UpdateYaml(self.yaml)

    def __EvaluateSoftAssign(self):
        """
        获取H, 计算软分配的聚类指标
        """
        q = self.cluster_model.degradation.get_q(self.cluster_model.H)
        self.y_predcit = torch.argmax(q, 1).cpu().numpy()
        # 软聚类的结果
        acc, nmi, ri, f_score = Metric.GetMetrics(self.y_true, self.y_predcit)
        self.soft_assign_loss['acc'] = acc
        self.soft_assign_loss['nmi'] = nmi
        self.soft_assign_loss['ri'] = ri
        self.soft_assign_loss['f_score'] = f_score 
        output = "Soft Assignment.  ACC:{:.4f}, NMI:{:.4f}, RI:{:.4f}, F_score:{:.4f}" \
            .format(acc, nmi, ri, f_score)
        print(output)
    
    def __StimulateCluster(self):
        """
        模拟多次聚类, 计算聚类后个指标的平均值
        """
        cluster = Cluster(self.cluster_model, self.cluster_dataset.dataset.y)
        acc, nmi, ri, f_score = cluster.ConductCluster()
        self.soft_assign_loss['acc'] = acc
        self.soft_assign_loss['nmi'] = nmi
        self.soft_assign_loss['ri'] = ri
        self.soft_assign_loss['f_score'] = f_score
        output = "Cluster Average. ACC:{:.4f}, NMI:{:.4f}, RI:{:.4f}, F-score:{:.4f}" \
             .format(acc, nmi, ri, f_score)
        print(output)

    def __SaveYaml(self):
        path = "description.yaml"
        if self.save_model:
            FileTool.SaveConfigYaml(os.path.join(self.model_path, path), self.yaml)
        if self.save_log:
            FileTool.SaveConfigYaml(os.path.join(self.log_path, path), self.yaml)

    def __SaveClusterReuslt(self):
        cluster_result = {
             # matlab和python中的维度是反的
            'y_true': self.y_true.reshape(-1,1),  # 这样有助于matlab中阅览
            'y_predict':self.y_predcit.reshape(-1,1)
        }
        file_name = "cluster_result.mat"
        scipy.io.savemat(os.path.join(self.model_path, file_name), cluster_result)


                
    def __SaveResult(self):
        """
        保存结果, 包括.yaml说明文件,误差文件，聚类结果,混淆矩阵图片
        """
        self.__SaveYaml()
        self.__SaveClusterReuslt()


    def StartTrain(self):
        if self.save_model == True:
            self.__SetFolder(self.model_path)
        if self.save_log == True:
            self.__SetFolder(self.log_path)
        self.__InitialYaml()

        # 后面的不训练，那么前面的也不应该训练
        if self.is_second_train == False:  
            self.is_first_train = False
        if self.is_first_train == False:
            self.is_pre_train = False
        self.__PreTrain()
        self.__FirstTrain()
        self.__SecondTrain()
        self.__EvaluateSoftAssign()
        # self.__StimulateCluster()
        self.__SaveResult()


        
 
    