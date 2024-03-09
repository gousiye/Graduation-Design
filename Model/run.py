import argparse
import numpy as np
from models import *    #导入的时候就执行了models/__init__.py
from utils import MyDataset, ClusterDataset, FileTool
import torch
import test
from train import Train

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
    myDataset = MyDataset(config['data_params']['data_path'], config['data_params']['data_name'])
    data_len = myDataset.GetLen()
    H = torch.from_numpy(np.random.uniform(0, 1, [data_len, 10])).float()
    myDataset.BindH(H) 
    
    encoders_decoders = ClusterModel.GenerateEncoders(
        myDataset.GetViewDims(), 
        config['model_params']['latent_encoder_dim'], 
        myDataset.GetClusterNum(), 
        config['pre_trainer_params']['devices'][0],
        test.encoder_code_list, 
        test.decoder_code_list
    )

    degradation = ClusterModel.GenerateDegradation(
        config['data_params']['batch_size'], 
        myDataset.GetClusterNum(), 
        test.degrade_code,
        **config['model_params']
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
    
    

    # # 需要预训练AE模型
    # if config['train_params']['is_pre_train'] == True:
        
    #     clusterDataset =  ClusterDataset(myDataset,config['data_params'])
    #     pretrain = PreTrain(
    #         cluster_model,
    #         config['data_params']['data_name'],
    #         config['exp_params']['lr_pre'],
    #         config['data_params']['cluster_num'],
    #         config['data_params']['view_num'],
    #         config['model_params']['save_pre_model'],
    #         config['model_params']['pre_model_path']
    #     )

    #     runner = Trainer(**config['pre_trainer_params'])
    #     runner.fit(pretrain, clusterDataset)

    #     # 保存训练的模型
    #     if pretrain.is_save == True:
    #         path = os.path.join(pretrain.save_path,pretrain.data_name) +'.pth.tar'
    #         FileTools.SaveModel(path, cluster_model.encoder_decoder)
    #         print("------------------------")
    #         print('\033[94m' + "已保存预训练AE模型到{path}" + '\033[0m'.format(path = path))
    #         print("------------------------")

    # else:   
    #     path = os.path.join(config['model_params']['pre_model_path'], config['data_params']['data_name']) +'.pth.tar'
    #     assert os.path.exists(path), '改模型没有存储文件'
    #     FileTools.LoadModel(path, cluster_model)
    #     print("------------------------")
    #     print('\033[94m' + "已从{path}读取预训练AE模型" + '\033[0m'.format(path = path))
    #     print("------------------------")


