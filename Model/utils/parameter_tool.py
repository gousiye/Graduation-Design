import copy
from typing import List,  Union, Any, TypeVar, Tuple, Dict
from models.submodels import EncoderDecoder
from models import ClusterModel
from torch import Tensor
class ParameterTool:    

    """"
    静态的超参处理工具类，不能实例化
    """
    def __init__(cls):
        raise TypeError("这是一个静态工具类，不能被实例化")
    
    @staticmethod
    def GetDescription(config, field) -> dict:
        pre_description = {}
        for para_aspect in field:
            pre_description[para_aspect] = {}
            for index, param in enumerate(field[para_aspect]):
               pre_description[para_aspect][param] = config[para_aspect][field[para_aspect][index]]
        return pre_description

    @staticmethod
    def __GetEncoderDescription(encoder_decoder:List[EncoderDecoder]) -> dict:
        description = {}
        iter_description= {}
        for i in  range(len(encoder_decoder)):
            coder_description = {'encoder':[], 'decoder':[]}
            for encoder in encoder_decoder[i].encoders:
                coder_description['encoder'].append(str(encoder))
            for decoder in encoder_decoder[i].decoders:
                coder_description['decoder'].append(str(decoder))
            index = 'view[{i}]'.format(i = i)
            iter_description[index] = coder_description
        description['encoder_decoders'] = iter_description       
        return description


    @staticmethod
    def GetModelDescription(cluster_model: ClusterModel) -> dict:
        """
        完整模型的版本
        model_structure:
            encoder_decoders:
                [0]:
                    encoder: Linear()
                             ReLU
                    decoder: Linear()
            degradation: 
                [0]:
                    Linear()
                    ReLU()
                    Linear()
                [1]:
                    
        """   
        description = {}
        inner_description = ParameterTool.__GetEncoderDescription(cluster_model.encoder_decoder)
        inner_description['degradation'] = {}
        for i in range(len(cluster_model.degradation.degrader)):
            index = 'view[{i}]'.format(i = i)
            inner_description['degradation'][index] = []
            for layer in cluster_model.degradation.degrader[i]:
                inner_description['degradation'][index].append(str(layer))
        description['model_structure'] = inner_description
        return description
        

    @staticmethod
    def GetPreModelDescription(encoder_decoder:List[EncoderDecoder]) -> dict:
        """
        预训练模型的版本
        model_structure:
            encoder_decoders:
                [0]:
                    encoder: Linear()
                             ReLU
                    decoder: Linear()
        """   
        encoders_description = ParameterTool.__GetEncoderDescription(encoder_decoder)
        descripition = {'model_structure':encoders_description}
        return descripition

        
    @staticmethod
    def GetPreLossDescription(ae_pre_loss: Tensor, view_loss:List[Tensor]):
        ae_pre_loss = ae_pre_loss.item()
        descripition = {}
        loss_description = {}
        loss_description['ae_pre_loss'] = ae_pre_loss
        view_description = {}
        for i in range(len(view_loss)):
            view_description['view[{num}]'.format(num = i)] = view_loss[i].item()
        loss_description['view_loss'] = view_description
        descripition['loss'] =loss_description
        return descripition
    
    @staticmethod
    def InitVarFromDict(obj:object, config:dict)->None:
        """
        把config.yaml中的参数转为对象的成员
        """
        for param_aspect in config:
            for parameter in config[param_aspect]:
                setattr(obj, parameter, config[param_aspect][parameter])