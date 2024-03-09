from torch import nn 
from abc import abstractmethod
from typing import List,  Union, Any, TypeVar, Tuple
from torch import Tensor

class BaseModel(nn.Module):

    def __int__ (self) -> None:
        super(BaseModel, self).__int__()

    #@abstractmethod
    def Encode(self, input:List[Tensor]) -> List[Tensor]:
        pass
    
    #@abstractmethod
    def Decode(self, input:Tensor)->Any:
        pass
    
    #@abstractmethod
    def Degradation(self, input:Tensor)->Any:
        pass
    
    #@abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    #@abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass
