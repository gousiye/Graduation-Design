from torch import Tensor
import torch.nn as nn
from typing import List

class Metric:
    """"
    静态的指标工具类，不能实例化
    """
    def __init__(cls):
        raise TypeError("这是一个静态工具类，不能被实例化")

    @staticmethod
    def GetAverageMSE(dataA: List[Tensor], dataB: List[Tensor]) -> Tensor:
        assert len(dataA) == len(dataB), '两个数据的长度不匹配, 无法计算MSE'
        tmp = 0
        dataLen = len(dataA)
        for i in range(dataLen):
            tmp = tmp + (nn.MSELoss()(dataA[i], dataB[i])) / dataLen
        return tmp            