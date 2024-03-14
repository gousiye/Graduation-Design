from torch import Tensor
import torch.nn as nn
from typing import List
import numpy as np


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

    @staticmethod
    def ACC(y_true:np.ndarray, y_pred:np.ndarray) -> float:
        """
        计算ACC
        """
        assert y_true.shape == y_pred.shape, '预测数据和真实数据维度不符'
        y_true = y_true.astype(np.int64)
        dim = max(y_true.max(), y_pred.max()) + 1
        matric = np.zeros((dim, dim), dtype=np.int64)
        for i in range(y_true.shape[0]):
            matric[y_pred[i]][y_true[i]] += 1   # 相当于混淆矩阵
        from scipy.optimize import linear_sum_assignment as linear_assignment
        row_index, col_index = linear_assignment(matric.max() - matric)
        return sum([matric[i][j] for i, j in zip(row_index, col_index)]) * 1.0 / y_pred.size


