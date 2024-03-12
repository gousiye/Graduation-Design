import argparse
import numpy as np
from models import *    #导入的时候就执行了models/__init__.py
from utils import MyDataset, ClusterDataset, FileTool
import torch
import test
from train import Train
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

parser = argparse.ArgumentParser(description='Deep Cluster Model')

parser.add_argument('--data', '-d', 
                    dest = 'data',
                    metavar = 'DATA',
                    help = 'select the data',
                    default= 'shuffled_Scene_2views')

# 不指定.yaml文件，则默认为和data同名的.yaml文件
parser.add_argument('--config',  '-c',    # 长格式--config, 短格式-c
                    dest="filename",    # 存储在args.filename
                    metavar='FILE',      # 使用说明中是 “文件类型”
                    help =  'path to the config file',
                    default='')
# 解析命令行
args = parser.parse_args()


def ConstructModelAndDataset():
    assert len(test.encoder_code_list) == len(test.decoder_code_list) and \
            len(test.encoder_code_list) == len(test.degrade_code), '编码器,解码器,退化网络的视角不匹配'
    myDataset = MyDataset(
        config['data_params']['data_path'], 
        config['data_params']['data_name'],
        config['model_params']['H_dim']
    )
    
    encoders_decoders = ClusterModel.GenerateEncoders(
        myDataset.GetViewDims(), 
        config['model_params']['latent_encoder_dim'], 
        myDataset.GetClusterNum(), 
        test.encoder_code_list, 
        test.decoder_code_list
    )

    degradation = ClusterModel.GenerateDegradation(
        config['data_params']['batch_size'], 
        config['model_params']['latent_encoder_dim'],
        config['model_params']['H_dim'],
        myDataset.GetClusterNum(),
        test.degrade_code,
    )
    cluster_model = ClusterModel(encoders_decoders, degradation)
    cluster_dataset =  ClusterDataset(myDataset,config['data_params'])
    return cluster_model, cluster_dataset


if __name__ == '__main__':
    config = FileTool.ReadConfig(args)
    print('--------------------start------------------------')
    print()
    cluster_model, cluster_dataset = ConstructModelAndDataset()
    
    train = Train(cluster_model, cluster_dataset, config)
    trainer = Trainer(
        logger = False,
        callbacks=[ModelCheckpoint(save_last=False, save_top_k=0, monitor=None)],
        accelerator = config['trainer_params']['accelerator'],
        devices = config['trainer_params']['devices'],
        max_epochs = config['trainer_params']['total_max_epochs'],  
    )
    print(cluster_dataset.dataset.H)
    trainer.fit(train, cluster_dataset)
    print(cluster_dataset.dataset.H)
    # print(cluster_model.degradation.degrader[0])