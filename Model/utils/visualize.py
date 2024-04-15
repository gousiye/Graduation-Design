import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
import seaborn 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
class Visualize:
    """
    静态的可视化工具类，不能实例化
    """
    def __init__(cls):
        raise TypeError("这是一个静态工具类, 不能被实例化")
    
    @staticmethod
    def GenerateConfuseGraph(
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        save_path: str
    ):
        """
        基于匈牙利算法，生成调整后的混淆矩阵
        """
        if save_path is None:
            return
        assert y_true.shape == y_pred.shape, "预测样本数和实际样本数不相符"
        y_true = y_true.astype(np.int64)
        y_pred = y_pred.astype(np.int64)
        O = max(y_pred.max(), y_true.max()) + 1
        M = np.zeros((O, O), dtype=np.int64)
        confuse_matrix = np.zeros((O, O), dtype=np.int64)  # 混淆矩阵
        for i in range(y_pred.size):
            M[y_pred[i], y_true[i]] += 1  # 纵轴是真实值，横轴是预测值
        # row_index: 0, 1, 2 , ...  # col_index: a1 ,a2 ,a3, ...
        row_index, col_index = linear_assignment(M.max() - M)
        # 构建初步的混淆矩阵
        for i , j in zip(row_index, col_index):
            confuse_matrix[:, i] = M[:, j]  # 预测类别row_index的实际和预测类别col_index匹配。
        confuse_matrix_norm = confuse_matrix.astype(float) / confuse_matrix.sum(axis = 1)[:, np.newaxis]  # 归一化
        confuse_matrix_norm = np.around(confuse_matrix_norm, 2)  # 保留到两位小数
        plt.figure(figsize=(12.7, 7)) # 大致维持1960/1080
        confusion_matrix_plot = seaborn.heatmap(confuse_matrix_norm, vmin=0, vmax=1, annot=True)

        confusion_matrix_plot.set_xlabel("true",fontsize = 20)
        confusion_matrix_plot.set_ylabel("predict", fontsize = 20)
        plt.savefig(save_path, dpi=1024)

    @staticmethod
    def Generate_TENE_H(y_true:np.ndarray, H:np.ndarray, save_path:str):
        """
        公共特征H二维可视化
        """
        y_true = y_true.astype(int)
        tsne = TSNE(n_components=2, init='pca', random_state = 100)
        X_tsne = tsne.fit_transform(H)
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
        plt.figure(figsize=(12.7, 7))
        for i in range(X_norm.shape[0]):
            plt.text(X_norm[i, 0], X_norm[i, 1], str(y_true[i]), color=plt.cm.Set1(y_true[i]),
                    fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        plt.savefig(save_path, dpi=1024)