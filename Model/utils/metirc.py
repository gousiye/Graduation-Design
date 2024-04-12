from torch import Tensor
import torch.nn as nn
from typing import List
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster._supervised import check_clusterings
from scipy.special import comb
from scipy import sparse as sp

class MetricTool:
    """
    用于计算指标的工具类, 静态类, 不能被实例化
    """
    def __init__(cls):
        raise TypeError("这是一个静态工具类，不能被实例化")


    @staticmethod
    def contingency_matrix(labels_true, labels_pred, eps=None, sparse=False):
        """
        构造列联矩阵
        """
        if eps is not None and sparse:
            raise ValueError("Cannot set 'eps' when sparse=True")

        classes, class_idx = np.unique(labels_true, return_inverse=True)
        clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
        n_classes = classes.shape[0]
        n_clusters = clusters.shape[0]
        # Using coo_matrix to accelerate simple histogram calculation,
        # i.e. bins are consecutive integers
        # Currently, coo_matrix is faster than histogram2d for simple cases
        contingency = sp.coo_matrix((np.ones(class_idx.shape[0]),
                                    (class_idx, cluster_idx)),
                                    shape=(n_classes, n_clusters),
                                    dtype=np.int64)
        if sparse:
            contingency = contingency.tocsr()
            contingency.sum_duplicates()
        else:
            contingency = contingency.toarray()
            if eps is not None:
                # don't use += as contingency is integer
                contingency = contingency + eps
        return contingency


    @staticmethod
    def b3_precision_recall_fscore(labels_true, labels_pred):
        """Compute the B^3 variant of precision, recall and F-score.
            Parameters
            ----------
            :param labels_true: 1d array containing the ground truth cluster labels.
            :param labels_pred: 1d array containing the predicted cluster labels.
            Returns
            -------
            :return float precision: calculated precision
            :return float recall: calculated recall
            :return float f_score: calculated f_score
            Reference
            ---------
            Amigo, Enrique, et al. "A comparison of extrinsic clustering evaluation
            metrics based on formal constraints." Information retrieval 12.4
            (2009): 461-486.
            """
            # Check that labels_* are 1d arrays and have the same size

        labels_true, labels_pred = check_clusterings(labels_true, labels_pred)

        # Check that input given is not the empty set
        if labels_true.shape == (0,):
            raise ValueError(
                "input labels must not be empty.")

        # Compute P/R/F scores
        n_samples = len(labels_true)
        true_clusters = {}  # true cluster_id => set of sample indices
        pred_clusters = {}  # pred cluster_id => set of sample indices

        for i in range(n_samples):
            true_cluster_id = labels_true[i]
            pred_cluster_id = labels_pred[i]

            if true_cluster_id not in true_clusters:
                true_clusters[true_cluster_id] = set()
            if pred_cluster_id not in pred_clusters:
                pred_clusters[pred_cluster_id] = set()

            true_clusters[true_cluster_id].add(i)
            pred_clusters[pred_cluster_id].add(i)

        for cluster_id, cluster in true_clusters.items():
            true_clusters[cluster_id] = frozenset(cluster)
        for cluster_id, cluster in pred_clusters.items():
            pred_clusters[cluster_id] = frozenset(cluster)

        precision = 0.0
        recall = 0.0

        intersections = {}

        for i in range(n_samples):
            pred_cluster_i = pred_clusters[labels_pred[i]]
            true_cluster_i = true_clusters[labels_true[i]]

            if (pred_cluster_i, true_cluster_i) in intersections:
                intersection = intersections[(pred_cluster_i, true_cluster_i)]
            else:
                intersection = pred_cluster_i.intersection(true_cluster_i)
                intersections[(pred_cluster_i, true_cluster_i)] = intersection

            precision += len(intersection) / len(pred_cluster_i)
            recall += len(intersection) / len(true_cluster_i)

        precision /= n_samples
        recall /= n_samples

        f_score = 2 * precision * recall / (precision + recall)

        return precision, recall, f_score


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
        y_true = y_true.astype(np.int64)
        assert y_pred.size == y_true.size
        O = max(y_pred.max(), y_true.max()) + 1
        M = np.zeros((O, O), dtype=np.int64)
        for i in range(y_pred.size):
            M[y_pred[i], y_true[i]] += 1
        row_index, col_index = linear_assignment(M.max() - M)
        return sum([M[i, j] for i, j in zip(row_index, col_index)]) * 1.0 / y_pred.size

    @staticmethod
    def NMI(y_true:np.ndarray, y_pred:np.ndarray) -> float:
        """
        计算NMI
        """
        return normalized_mutual_info_score(y_true, y_pred)
    
    @staticmethod
    def RI(labels_true:np.ndarray, labels_pred:np.ndarray) -> float:
        """
        计算RI
        """
        labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
        n_samples = labels_true.shape[0]
        n_classes = np.unique(labels_true).shape[0]
        n_clusters = np.unique(labels_pred).shape[0]
        # Special limit cases: no clustering since the data is not split;
        # or trivial clustering where each document is assigned a unique cluster.
        # These are perfect matches hence return 1.0.
        if (n_classes == n_clusters == 1 or
                n_classes == n_clusters == 0 or
                n_classes == n_clusters == n_samples):
            return 1.0

        # Compute the RI using the contingency data
        contingency = MetricTool.contingency_matrix(labels_true, labels_pred)

        n = np.sum(np.sum(contingency))
        t1 = comb(n, 2)
        t2 = np.sum(np.sum(np.power(contingency, 2)))
        nis = np.sum(np.power(np.sum(contingency, 0), 2))
        njs = np.sum(np.power(np.sum(contingency, 1), 2))
        t3 = 0.5 * (nis + njs)

        A = t1 + t2 - t3
        nc = (n * (n ** 2 + 1) - (n + 1) * nis - (n + 1) * njs + 2 * (nis * njs) / n) / (2 * (n - 1))
        AR = (A - nc) / (t1 - nc)
        return A / t1
    
    @staticmethod
    def F_score(labels_true, labels_pred):
        """Compute the B^3 variant of F-score.
        Parameters
        ----------
        :param labels_true: 1d array containing the ground truth cluster labels.
        :param labels_pred: 1d array containing the predicted cluster labels.
        Returns
        -------
        :return float f_score: calculated F-score
        """
        _, _, f = MetricTool.b3_precision_recall_fscore(labels_true, labels_pred)
        return f
    
    @staticmethod
    def GetMetrics(labels_true: np.ndarray, labels_pred: np.ndarray):
        acc =  Metric.ACC(labels_true, labels_pred) 
        nmi = Metric.NMI(labels_true, labels_pred)
        ri = Metric.RI(labels_true, labels_pred) 
        f_score = Metric.F_score(labels_true, labels_pred)
        return acc, nmi, ri, f_score
    

    @staticmethod
    def GetAvgMetrics(labels_true: np.ndarray, labels_pred: List[np.ndarray]):
        cnt = len(labels_pred) # 重复的次数
        acc_array = np.zeros(cnt)
        nmi_array = np.zeros(cnt)
        ri_array = np.zeros(cnt)
        f_score_array = np.zeros(cnt)
        for i in range(cnt):
            acc_array[i], nmi_array[i], ri_array[i], f_score_array[i] \
                = Metric.GetMetrics(labels_true, labels_pred[i])
        acc_avg = np.mean(acc_array)
        nmi_avg = np.mean(nmi_array)
        ri_avg = np.mean(ri_array)
        f_score_avg = np.mean(f_score_array)
        return acc_avg, nmi_avg, ri_avg, f_score_avg
    