import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from typing import List,  Union, Any, TypeVar, Tuple, Dict
from .submodels import *
from collections import defaultdict
import numpy as np

class ClusterModel(torch.nn.Module):

    @staticmethod
    def GenerateEncoders(
        view_feature_dim_list: List[int], 
        latent_encoder_dim:int, 
        cluster_num:int,
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
                encode_code_list[i], 
                decode_code_list[i]
            )
            result.append(tempEncoder)
        return result
    
    @staticmethod
    def GenerateDegradation(
        batch_size:int, 
        latent_encoder_dim:int, 
        H_dim:int, 
        cluster_num: int, 
        code: List[str]
    ):
        degradation = Degradation(
            batch_size, 
            latent_encoder_dim,
            H_dim,
            cluster_num, 
            code
        )
        return degradation
        

    def __init__(
        self, 
        encoder_decoder: List[EncoderDecoder], 
        degradation: Degradation,
        **kwargs:dict
    ) -> None:
        super(ClusterModel, self).__init__()
        self.encoder_decoder = encoder_decoder #编码器，解码器
        self.degradation = degradation
        self.H  = None
        self.cluster_num = self.degradation.cluster_num  
        for iter in self.encoder_decoder:
            assert self.cluster_num == iter.cluster_num, "编码器和退化网络的类别数量不一致"

    def get_z_half_list(self, features:List[Tensor]) -> List[Tensor]:
        result = [None] * len(self.encoder_decoder)
        for i in range(len(self.encoder_decoder)):        
            result[i] = self.encoder_decoder[i].get_z_half(features[i])
        return result

    def get_ae_recon_list(self, features:List[Tensor]) -> List[Tensor]:
        result = [None] * len(self.encoder_decoder)
        for i in range(len(self.encoder_decoder)):        
            result[i] = self.encoder_decoder[i](features[i])
        return result
    
    def get_ae_q_list(self, features:List[Tensor])->List[Tensor]:
        result = [None] * len(self.encoder_decoder)
        for i in range(len(self.encoder_decoder)):
            result[i] = self.encoder_decoder[i].get_q(features[i]) 
        return result

    def GenerateH(self, length, dim):
        self.H = torch.from_numpy(np.random.uniform(0, 1, [length, dim])).float()

    def forward(self):
        # return self
        pass

    # def __GetSoftClusterMetric(self):
    #     """
    #     根据软聚类分配直接得到结果, 选择概率最大的那个类别
    #     """       
    #     batch_y = []
    #     final_h = self.cluster_model.H.to(f'cuda:{self.devices[0]}')
    #     final_q = self.cluster_model.degradation.get_q(final_h) 
    #     for _, labels in self.dataloader:
    #         batch_y.append(labels)
    #     final_y = torch.cat(batch_y).cpu().numpy()
    #     final_result = torch.argmax(final_q, 1).cpu().numpy()
    #     acc, nmi, ri, f_score = Metric.GetMetrics(final_y, final_result)
    #     return acc, nmi, ri, f_score

    


if __name__ == "__main__":
    pass        