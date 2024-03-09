from torch import nn 
from typing import List,  Union, Any, TypeVar, Tuple
from torch import Tensor
import torch

class Degradation(nn.Module):
    def __init__(self):
        super(Degradation, self).__init__()
        pass

    def __init__(
        self, 
        batch_size:int, 
        latent_encoder_dim:int, 
        h_dim:int,
        cluster_num: int, code:str
    ):
        super(Degradation, self).__init__()
        self.v = torch.nn.Parameter(torch.FloatTensor(batch_size, h_dim))
        self.center = torch.nn.Parameter(torch.FloatTensor(cluster_num, latent_encoder_dim))  # k x d 
        self.alpha = 1
        
    def forward(self):
        pass
if __name__ == "__main__":
    degradation = Degradation(32, 128, 128, 10, 'ss')