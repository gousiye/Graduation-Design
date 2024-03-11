from torch import nn 
from abc import abstractmethod
from typing import List,  Union, Any, TypeVar, Tuple
from torch import Tensor
import torch
import torch.nn as nn

class BaseModel(nn.Module):

    def __int__ (self) -> None:
        super(BaseModel, self).__int__()
    
    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    # 初始化线性层的系数，偏置
    def InitCoefficient(self, model:torch.nn.Module) -> None:
        for iter in model:
            if isinstance(iter, nn.Linear):
                torch.nn.init.xavier_uniform_(iter.weight)
                torch.nn.init.constant_(iter.bias, 0)