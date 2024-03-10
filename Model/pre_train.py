import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from typing import Tuple
import torch.nn as nn
from models.submodels import EncoderDecoder
from typing import List, Optional, Sequence, Union, Any, Callable, Dict
import torch
from torch import Tensor
from torch import optim

class PreTrain(pl.LightningModule):

    def __init__(
        self, 
        encoder_decoder: List[EncoderDecoder],
        lr: float,
        cluster_num: int,
        description: dict
    ) -> None:
        
        super(PreTrain, self).__init__()
        self.encoder_decoder = encoder_decoder
        self.lr = lr
        self.cluster_num = cluster_num
        self.description = description
        self.loss = [0]* len(self.encoder_decoder)
        self.min_loss = float('inf')
        self.best_preModel_state = [None] * len(self.encoder_decoder)
        self.best_epoch_idx = -1

    def on_train_start(self) -> None:
        print()
        print("----------------------预训练开始---------------------")
        print()
        

    # pytorch_lightning 只要在training_step中正确计算，优化器参数列表包含就可以。这样也是可以的
    def forward(self, features):
        ae_pre_loss_tmp = 0
        for i in range(len(features)):
            tmpMSE = nn.MSELoss()(features[i], self.encoder_decoder[i](features[i]))
            self.loss[i] += tmpMSE
            ae_pre_loss_tmp += tmpMSE
        return ae_pre_loss_tmp

    def training_step(self, batch:Tuple, batch_idx:int):
        features, _, _ = batch
        ae_pre_loss = self.forward(features)
        self.log('ae_pre_loss', ae_pre_loss)
        return {'loss':ae_pre_loss}
    
    def on_train_epoch_start(self) -> None:
        for i in range(len(self.loss)):
            self.loss[i] = 0

    def training_epoch_end(self, outputs: List) -> None:
        current_epoch = self.current_epoch
        # tensor.mean 默认会计算全局平均
        ae_pre_loss = torch.stack([x['loss'] for x in outputs]).mean()
        print()
        output = "Pre_epoch: {:.0f}, Mean: ae_pre_loss: {:.4f}.  Sum: ".format(current_epoch, ae_pre_loss)
        for i in range(len(self.loss)):
            if i != len(self.loss) - 1:
                output += "loss{:.0f} = {:.4f}, ".format(i , self.loss[i])
            else:
                 output += "loss{:.0f} = {:.4f}.".format(i , self.loss[i])
        print(output)

        # 保存最优的模型
        if self.best_epoch_idx == -1 or ae_pre_loss < self.min_loss:
            self.best_epoch_idx = current_epoch
            self.min_loss = ae_pre_loss
            self.SaveBetterModel()
    
    def on_train_end(self):
        print()
        print("最好的预训练模型在第{i}个epoch, 其ae_pre_loss为{loss:.4f}".format(i = self.best_epoch_idx, loss = self.min_loss) )
        print("----------------------预训练结束---------------------")
        print()
        for i in range(len(self.encoder_decoder)):
            self.encoder_decoder[i].load_state_dict(self.best_preModel_state[i])
            pass

    def configure_optimizers(self):
        params = []
        for iter in self.encoder_decoder:
            params.extend(iter.parameters())
        return optim.Adam(params, lr = 0.001)

    def SaveBetterModel(self):
        for i in range(len(self.encoder_decoder)):
            self.best_preModel_state[i] = self.encoder_decoder[i].state_dict()




if __name__ == '__main__':
    pass