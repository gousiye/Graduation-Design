import os
import yaml
from pathlib import Path
from models import ClusterModel
from models.submodels import EncoderDecoder, Degradation
from typing import List, Optional, Sequence, Union, Any, Callable
import torch
import copy
import datetime
import utils 

class FileTool:
    """"
    静态的文件工具类，不能实例化
    """
    def __init__(cls):
        raise TypeError("这是一个静态工具类，不能被实例化")
    

    @staticmethod
    def ReadConfig(args) -> dict:
        """
        根据命令行的参数读取响应的配置文件
        """
        config = {}
        if args.filename == '':
            args.filename = os.path.join('configs', args.data) + '.yaml'
        with open(args.filename, 'r') as config_file:
            try:
                config = yaml.safe_load(config_file)
            except yaml.YAMLError as error :
                print(error)
            assert config['data_params']['data_name'] == args.data, '配置文件和指定的数据集不相符'
            return config
    

    @staticmethod
    def SaveConfigYaml(
        path: str,
        config: str
    ) -> dict:        
        """
        生成该模型的描述文件description.yaml
        """
        with open(path, 'w', encoding='utf-8') as file:
            yaml.dump(config, file, sort_keys=False)



    @staticmethod
    def __CacheAEModel(
        encoder_list: List[EncoderDecoder], 
        loss: dict = None, 
    ) -> dict:
        """
        保存AE模型到字典中
        """
        save_dict = {}
        save_dict['view_num'] = len(encoder_list)
        for i in range(len(encoder_list)):
            model_key = "AE{num}".format(num = i)
            save_dict[model_key] = encoder_list[i].state_dict()
        if loss is not None:
            save_dict['loss'] = {}
            for key in loss:
                save_dict['loss'][key] = loss[key]
        return save_dict


    @staticmethod
    def __ReadAEModel(
        checkPoint:dict,
        encoder_decoder: List[EncoderDecoder], 
        loss:dict,
    ) -> tuple:
        """
        读取AE模型
        """
        loss.clear()
        encoder_list = [None] * checkPoint['view_num']
        assert len(encoder_list) == len(encoder_decoder), "模型与读取模型的视角数不匹配"
        for i in range(len(encoder_list)):
            model_key = "AE{num}".format(num = i)
            encoder_decoder[i].load_state_dict(checkPoint[model_key], False)
        if loss is not None:
            for key in checkPoint['loss']:
                loss[key] = checkPoint['loss'][key]



    @staticmethod
    def SavePreModel(
        path:str,
        encoder_list :List[EncoderDecoder],
        loss: dict = None,
    )->None:
        save_dic = FileTool.__CacheAEModel(encoder_list, loss)
        torch.save(save_dic, path)


    @staticmethod
    def LoadPreModel(
        path:str,
        encoder_decoder :List[EncoderDecoder],
        loss: dict = None,
    )->None:
        assert path is not None, "读取路径为空"
        assert os.path.exists(path), "改路径不存在模型文件"
        checkPoint = torch.load(path)
        FileTool.__ReadAEModel(checkPoint, encoder_decoder,loss)


    @staticmethod
    def __SaveModel(
        path: str, 
        cluster_model: ClusterModel,
        loss_parameters: dict
    ):
        save_dict = FileTool.__CacheAEModel(cluster_model.encoder_decoder, loss_parameters)
        save_dict['Degradation'] = cluster_model.degradation.state_dict()
        save_dict['H'] = cluster_model.H
        torch.save(save_dict, path)

    @staticmethod
    def __LoadModel(
        path:str,
        cluster_model: ClusterModel,
        loss_parameters:dict
    ):
        assert path is not None, "读取路径为空"
        assert os.path.exists(path), "改路径不存在模型文件"
        checkPoint = torch.load(path)
        FileTool.__ReadAEModel(checkPoint, cluster_model.encoder_decoder, loss_parameters)
        cluster_model.degradation.load_state_dict(checkPoint['Degradation'])
        cluster_model.H = checkPoint['H']

    @staticmethod
    def SaveModelAndLog(
        cluster_model: ClusterModel, 
        config:dict,
        path:dict,
        loss_parameters:dict
    ):
        """
        正式训练保存模型及日志
        """
        model_path = path['model_path']
        log_path = path['log_path']
        data_name = path['data_name']

        description = {}

        now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        check_folder_path = os.path.join(model_path, data_name)    
        check_model_path = os.path.join(check_folder_path, data_name) + '.pth.tar'
        check_param_path = os.path.join(check_folder_path, data_name) + '.yaml'

        log_folder_path = os.path.join(log_path, data_name + now_time)    
        log_model_path = os.path.join(log_folder_path, data_name) + '.pth.tar'
        log_param_path = os.path.join(log_folder_path, data_name) + '.yaml'

        hyper_parameters = config
        model_parameters = utils.ParameterTool.GetModelDescription(cluster_model)

        if not os.path.exists(check_folder_path):
            os.makedirs(check_folder_path)
        if not os.path.exists(log_folder_path):
            os.makedirs(log_folder_path)
        
        FileTool.__SaveModel(check_model_path, cluster_model, loss_parameters)
        FileTool.__SaveModel(log_model_path, cluster_model, loss_parameters)

        description.update(hyper_parameters)
        description.update(loss_parameters)
        description.update(model_parameters)

        FileTool.SaveConfigYaml(check_param_path, description)
        FileTool.SaveConfigYaml(log_param_path, description)
        


    @staticmethod 
    def LoadAndLog(
        cluster_model:ClusterModel,
        config:dict,
        path:dict
    ):
        """
        读取模型，日志
        """
        model_path = path['model_path']
        data_name = path['data_name']

        description = {}

        check_folder_path = os.path.join(model_path, data_name)    
        check_model_path = os.path.join(check_folder_path, data_name) + '.pth.tar'
        check_param_path = os.path.join(check_folder_path, data_name) + '.yaml'

        assert os.path.exists(check_model_path), '该模型没有存储文件'
        loss_parameters = {}

        FileTool.__LoadModel(check_model_path, cluster_model, loss_parameters)
        

        hyper_parameters = config
        model_parameters = utils.ParameterTool.GetModelDescription(cluster_model)

        description.update(hyper_parameters)
        description.update(loss_parameters)
        description.update(model_parameters)
        FileTool.SaveConfigYaml(
            check_param_path,
            description
        )
        


