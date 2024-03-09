import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from typing import List,  Union, Any, TypeVar, Tuple


# 自动编码解码器
class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()
        pass

    def __init__(
        self, 
        feature_dim:int, 
        latent_encoder_dim:int, 
        cluster_num:int, 
        device_num: int, 
        encoder_code:str, 
        decoder_code:str
    )->None:
        super(EncoderDecoder, self).__init__()

        self.feature_dim = feature_dim
        self.latent_encoder_dim = latent_encoder_dim
        self.device_num = device_num
        self.encoder_code = "self.encoders = " + encoder_code
        self.decoder_code = "self.decoders = " + decoder_code
        self.center = torch.nn.Parameter(torch.FloatTensor(cluster_num, latent_encoder_dim))  # k x d 
        # print(self.center.shape)
        self.alpha = 1
        exec(self.encoder_code)
        exec(self.decoder_code)

        self.encoders.to(f'cuda:{self.device_num}')
        self.decoders.to(f'cuda:{self.device_num}')

        self.InitCoefficient(self.encoders)
        self.InitCoefficient(self.decoders)

    # 初始化线性层的系数，偏置
    def InitCoefficient(self, model:torch.nn.Module) -> None:
        for iter in model:
            if isinstance(iter, nn.Linear):
                torch.nn.init.xavier_uniform_(iter.weight)
                torch.nn.init.constant_(iter.bias, 0)
    
    def forward(self, x: List[Tensor])-> List[Tensor]:
        x = x.to(f'cuda:{self.device_num}')
        z = self.get_z_half(x)
        x_reconstruct = self.decoders(z)
        return x_reconstruct

    def get_z_half(self, x: List[Tensor]) -> List[Tensor]:
        return self.encoders(x)
    
    # 计算聚类分配软分布
    def get_q(self, z):
        xe = torch.unsqueeze(z, 1) - self.center

    


if __name__ == "__main__":
    encoder_code = """nn.Sequential(
    nn.Linear(self.feature_dim, self.latent_encoder_dim),
    nn.ReLU(),
    nn.Sigmoid()
)"""
    decoder_code = """nn.Sequential(
    nn.Linear(self.latent_encoder_dim, self.feature_dim),
    nn.ReLU(),
)""" 
    instance = EncoderDecoder(20, 128, 10, encoder_code, decoder_code)
    # print(instance)
    # for iter in encoder.modules():
    #     print(iter)
