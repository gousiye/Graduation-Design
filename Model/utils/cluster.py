from models import ClusterModel
from utils import Metric
import numpy as np
from sklearn.cluster import KMeans

class Cluster():
    """
    给定一个聚类模型cluster_model。根据cluster_model中的H进行times次数的k-means聚类。
    计算各指标的平均值
    """
    def __init__(
            self, 
            cluster_model: ClusterModel, 
            y_true:np.ndarray,
            times:int = 10
    ):
        self.cluster_model = cluster_model
        self.y_true = y_true
        self.y_predict = [] #计算平均值用的
        self.times = times

    def ConductCluster(self):
        """
        在给定的cluter_model的H进行k-means聚类
        """       
        for i in range(self.times):
            km = KMeans(self.cluster_model.cluster_num, n_init= 10)
            predict = km.fit_predict(self.cluster_model.H)
            self.y_predict.append(predict)
        if np.min(self.y_true) == 1:
            self.y_true -= 1  # 默认0是第一类，与软分配一致
        acc_avg, nmi_avg, ri_avg, f_score_avg = Metric.GetAvgMetrics(self.y_true, self.y_predict)
        output = "Cluster Average. ACC:{:.4f}, NMI:{:.4f}, RI:{:.4f}, F-score:{:.4f}" \
            .format(acc_avg, nmi_avg, ri_avg, f_score_avg)
        print(output)