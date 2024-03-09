import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from typing import List,  Union, Any, TypeVar, Tuple, Dict
from .submodels import *
from collections import defaultdict

class ClusterModel(torch.nn.Module):

    @staticmethod
    def GenerateEncoders(
        view_feature_dim_list: List[int], 
        latent_encoder_dim:int, 
        cluster_num:int,
        device_num: int,
        encode_code_list:List[str], 
        decode_code_list:List[str],
    ):
        assert len(view_feature_dim_list) == len(encode_code_list) and len(decode_code_list) == len(decode_code_list), '视角数不匹配'
        result = []
        for i in range(len(view_feature_dim_list)):
            tempEncoder = EncoderDecoder(
                view_feature_dim_list[i], 
                latent_encoder_dim, 
                cluster_num,
                device_num, 
                encode_code_list[i], 
                decode_code_list[i]
            )
            result.append(tempEncoder)
        return result
    
    @staticmethod
    def GenerateDegradation(
        batch_size:int, 
        cluster_num: int, 
        code:str, 
        **kwargs
    ):
        degradation = Degradation(batch_size, kwargs['latent_encoder_dim'], kwargs['h_dim'], cluster_num, code)
        return degradation
        

    def __init__(
        self, 
        encoder_decoder: List[EncoderDecoder], 
        degradation: Degradation = None,
        **kwargs:dict
    ) -> None:
        super(ClusterModel, self).__init__()
        self.encoder_decoder = encoder_decoder #编码器，解码器
        self.degradation = degradation

    def forward(self):
        pass



if __name__ == "__main__":
    pass        