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
    def CutDescription(description: dict) -> dict:
        cutted_description = copy.deepcopy(description)
        redundant_field = [
            {'model_params': ['save_pre_model','pre_model_path','save_model', 'model_path']},
            {'data_params':['data_path']},
            {'exp_params':['lr_ae', 'lr_dg', 'lr_h']},
            {'train_params':['is_pre_train', 'is_pre_H']},
            {'log_params':['pre_log_path','log_path']}
        ]
        for dict in redundant_field:
            para_aspect = next(iter(dict))
            for para in dict[para_aspect]:
                del cutted_description[para_aspect][para]
            if len(cutted_description[para_aspect]) == 0:
                del cutted_description[para_aspect]

        return cutted_description

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
            iter_description['[{i}]'.format(i = i)] = coder_description
        description['encoder_decoders'] = iter_description       
        return description


    @staticmethod
    def GetModelDescription(cluster_model: ClusterModel, is_pre: bool) -> dict:
        """
        model_structure:
            encoder_decoders:
                [0]:
                    encoder: Linear()
                             ReLU
                    decoder: Linear()
            degradation: Linear()
        """   
        encoders_description = ParameterTool.__GetEncoderDescription(cluster_model.encoder_decoder)
        descripition = {'model_structure':encoders_description}
        if(is_pre == True):
            return descripition
        else:
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