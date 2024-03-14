import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from typing import List,  Union, Any, TypeVar, Tuple
from models.submodels import BaseModel

# 自动编码解码器
class EncoderDecoder(BaseModel):
    def __init__(self):
        super(EncoderDecoder, self).__init__()
        pass

    def __init__(
        self, 
        feature_dim:int, 
        latent_encoder_dim:int, 
        cluster_num:int,  
        encoder_code:str, 
        decoder_code:str
    )->None:
        super(EncoderDecoder, self).__init__()

        self.feature_dim = feature_dim
        self.latent_encoder_dim = latent_encoder_dim
        self.encoder_code = "self.encoders = " + encoder_code
        self.decoder_code = "self.decoders = " + decoder_code
        self.center = torch.nn.Parameter(torch.FloatTensor(cluster_num, latent_encoder_dim))  # k x d 
        # print(self.center.shape)
        self.alpha = 1
        exec(self.encoder_code)
        exec(self.decoder_code)

        super().InitCoefficient(self.encoders)
        super().InitCoefficient(self.decoders)


    
    def forward(self, x: List[Tensor])-> List[Tensor]:
        z = self.get_z_half(x)
        x_reconstruct = self.decoders(z)
        return x_reconstruct

    def get_z_half(self, x: List[Tensor]) -> List[Tensor]:
        return self.encoders(x)
    
    def set_center(self, center: Tensor):
        self.center = torch.nn.Parameter(center)

    # 计算聚类分配软分布
    def get_q(self, z: Tensor) -> Tensor:
        """
        学生T分布获得软聚类分配
        """
        # 广播原则 (batch_size, h_dim) - (cluster_num, h_dim) = (batch_size, cluster_num, h_dim)
        xe = torch.unsqueeze(z, 1) - self.center 
        # 计算z_half 到 center质心距离的平方, distance(batch_size, cluster_num)
        distance = torch.sum(torch.mul(xe, xe), 2) / self.alpha
        # 计算单个项
        q = 1.0/(1.0 + distance)
        q = q ** (self.alpha + 1.0) / 2.0
        # 计算分母, denominator(batch_size)
        denominator =  torch.sum(q, 1)
        # 计算q
        q = (q.t() /denominator).t()
        # q:(batch_size, cluster_num), 表示属于第cluster_num类别的概率
        return q
    


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
