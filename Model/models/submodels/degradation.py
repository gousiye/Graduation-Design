from torch import nn 
from typing import List,  Union, Any, TypeVar, Tuple
from torch import Tensor
import torch
from models.submodels import BaseModel

class Degradation(BaseModel):
    def __init__(self):
        super(Degradation, self).__init__()
        pass

    def __init__(
        self, 
        batch_size:int, 
        latent_encoder_dim:int, 
        H_dim:int,
        cluster_num: int, 
        degrade_net_code:List[str]
    ):
        super(Degradation, self).__init__()
        self.h_dim = H_dim
        self.latent_encoder_dim = latent_encoder_dim
        self.h = torch.nn.Parameter(torch.FloatTensor(batch_size, H_dim))
        self.center = torch.nn.Parameter(torch.FloatTensor(cluster_num, latent_encoder_dim)) # k x d 
        self.alpha = 1
        self.degrader = torch.nn.ModuleList()
        for i in range(len(degrade_net_code)):
            degrade_code = 'self.degrader.append(' + degrade_net_code[i] +')'
            exec(degrade_code)
            super().InitCoefficient(self.degrader[i])

    def get_view_h_list(self) -> List[Tensor]:
        """
        将H在各视角进行编码, 解码, 得到每个视角的h
        """
        view_h = [None] * len(self.degrader)
        for i in range(len(self.degrader)):
            view_h[i] = self.degrader[i](self.h)
        return view_h

    def forward(self):
        return self.h
    
    def set_h(self, h:Tensor):
        """
        id(self.H) != id(H)
        """
        self.h = torch.nn.Parameter(h)

    def get_h(self):
        return self.h
    
    def set_center(self, center):
        self.center = torch.nn.Parameter(center)
    
    def get_q(self, z: Tensor):
        #print(z.shape)
        #print(self.center.shape)
        # t = torch.unsqueeze(z, 1)
        # print(t.shape)
        xe = torch.unsqueeze(z, 1) - self.center
        print(xe.shape)

if __name__ == "__main__":
    degradation = Degradation(32, 128, 128, 10, 'ss')