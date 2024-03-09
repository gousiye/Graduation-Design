# 读取.mat文件，进行数据集的构建
import os
from torch.utils.data import DataLoader, Dataset
import scipy.io
import h5py
import numpy as np
import torch
from torch import Tensor
from typing import List
from pytorch_lightning import LightningDataModule
from typing import List, Optional, Sequence, Union, Any, Callable
from sklearn.preprocessing import MinMaxScaler

class MyDataset(Dataset):
    def __init__(
        self, 
        data_path: str, 
        data_name: str,
    ):
        self.path = data_path
        self.filename = data_name
        self.data_path = os.path.join(self.path, self.filename) + '.mat'
        self.x = []
        self.y = []
        self.H = []
        self.view_dims = []
        try :
            dataset = scipy.io.loadmat(self.data_path)
            # scipy.io.loadmat(·)会加上三个key'__header__', '__version__', '__globals__'
            length = len(dataset) - len(['__header__', '__version__', '__globals__'])
            x_index_name = ['x' + str(i) for i in range(1, length)] # 获取视角的名称
            self.x = [dataset[item][()] for item in x_index_name]
            self.y = dataset['y'][()]  
            self.y = np.squeeze(self.y)


        except FileNotFoundError: # 指定.mat文件不存在
            raise FileNotFoundError
        except NotImplementedError:  # .mat在7.3版本之上，需要使用h5py进行读取
            dataset = h5py.File(self.data_path, mode='r')
            length = len(dataset)
            x_index_name = ['x' + str(i) for i in range(1, length)] # 获取视角的名称
            self.x = [dataset[item][()].transpose() for item in x_index_name]
            self.y = dataset['y'][()].transpose()
            self.y = np.squeeze(self.y)
            
        self.view_dims = [self.x[i].shape[1] for i in range(len(self.x))]
        self.Normalize()

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        features = []
        for view_idx in range(len(self.x)):
            features_view = self.x[view_idx][idx][:]
            features_view = torch.tensor(features_view, dtype=torch.float32)
            features.append(features_view)
        return features, self.y[idx], self.H[idx]

    def Normalize(self):
        for i in range(len(self.x)):
            scalar = MinMaxScaler((0,1))
            self.x[i] = scalar.fit_transform(self.x[i])

    def BindH(self ,H: Tensor) -> None:
        self.H = H

    def GetLen(self) -> int:
        return self.y.shape[0]

    # 获取每个视角特征的维度
    def GetViewDims(self)-> List[int]:
        return self.view_dims

    def GetClusterNum(self) -> int:
        return len(np.unique(self.y))
    


class ClusterDataset(LightningDataModule):
    def __init__(
        self, 
        dataset: Dataset,
        params: dict,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.batch_size = params['batch_size']
        self.num_workers = params['num_workers']
    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

# 测试MyDataset是否能正常使用
if __name__ == '__main__':
        
    print('-------start----------')
    myDataset = MyDataset('dataset', 'shuffled_Scene_2views')
    data_len = myDataset.GetLen()
    H = torch.from_numpy(np.random.uniform(0, 1, [data_len, 10])).float()
    myDataset.BindH(H)    
    print(type(myDataset.H))
    data_loader = DataLoader(myDataset, batch_size = 32)
    for feature, cluster, h in data_loader:
        print(h[-1])

    dic = {'batch_size': 32}
    clusterDataset = ClusterDataset(myDataset, dic)