import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import random_split
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy
import pandas as pd
from torchmetrics.functional import accuracy
from torch.optim.lr_scheduler import StepLR
import numpy as np
from torchvision import transforms, datasets
from torchvision.transforms import Compose
from torch.utils.data import ConcatDataset
import random
from torchvision.transforms import ToTensor
import itertools
import yaml
import csv
import argparse
#-----------------------------------------------------------------------------------------------
class CNN(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.layer1 = nn.Sequential(
      nn.Conv1d(1, 8, kernel_size= 7, stride=1, padding=0),
      nn.BatchNorm1d(8),
      nn.MaxPool1d(kernel_size=2, stride=2),
      nn.LeakyReLU())
    self.layer2 = nn.Sequential(
      nn.Conv1d(8, 8, kernel_size= 5, stride=1, padding=0),
      nn.BatchNorm1d(8),
      nn.MaxPool1d(kernel_size=2, stride=2),
      nn.LeakyReLU())
    self.layer3 = nn.Sequential(
      nn.Conv1d(8, 16, kernel_size= 3, stride = 1, padding = 0),
      nn.BatchNorm1d(16),
      nn.MaxPool1d(kernel_size=2, stride=2),
      nn.LeakyReLU())
    self.layer4 = nn.Sequential(
      nn.Conv1d(16, 16, kernel_size = 3, stride = 1, padding = 0),
      nn.BatchNorm1d(16),
      nn.MaxPool1d(kernel_size=2, stride=2),
      nn.LeakyReLU())
    self.layer5 = nn.Sequential(
       nn.Conv1d(16, 16, kernel_size= 3, stride= 1, padding= 0),
       nn.BatchNorm1d(16),
       nn.MaxPool1d(kernel_size=2, stride=2),
       nn.LeakyReLU())
    self.layer6 = nn.Sequential(
       nn.Conv1d(16, 16, kernel_size= 3, stride= 1, padding= 0),
       nn.BatchNorm1d(16),
       nn.MaxPool1d(kernel_size=4, stride=2),
       nn.LeakyReLU())
    # this layer is convolving the output with HIE level
    self.layer6_conv1 = nn.Sequential( 
      nn.Conv1d(2, 1, kernel_size = 1, stride = 1, padding = 0))
    # this layer is convolving the 16 channels together and reduces the dimension 
    self.layer6_conv2 = nn.Sequential(
      nn.Conv1d(16, 1, kernel_size= 1, stride= 1, padding= 0))

  def forward(self, x, hie):
    out = self.layer1(x)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = self.layer5(out)
    out = self.layer6(out)
    out = out.reshape(out.shape[0], out.shape[1])
    extended_hie = torch.repeat_interleave(hie, 16)
    extended_hie = extended_hie.reshape(hie.shape[0],-1)
    out = torch.cat((out, extended_hie), dim=0)
    out = out.reshape( hie.shape[0],2, 16)
    out = self.layer6_conv1(out)
    out = out.reshape( hie.shape[0],16, 1)
    out = self.layer6_conv2(out)
    return out


  def training_step(self, batch_loader, batch_idx):
    X, hie, Y = batch_loader
    Y_preds = self(X.float(), hie.float())
    Y_preds = Y_preds.reshape(-1,1)
    loss = loss_fn(Y_preds, Y.float())
    acc = accuracy(Y_preds, Y, task = "binary")
    self.log('train_loss', loss) 
    self.log('train_accuracy', acc)    
    return {'loss': loss, 'accuracy':acc} 
    
  
  def test_step(self, batch_loader, batch_idx):
    X, hie, Y = batch_loader
    Y_preds = self(X.float(), hie.float())
    Y_preds = Y_preds.reshape(-1,1)
    loss = loss_fn(Y_preds, Y.float())
    acc = accuracy(Y_preds, Y, task = "binary")
    self.log('test_accuracy', acc)
    self.log('test_loss', loss)
    return {'loss': loss, 'accuracy':acc} 
  
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
    scheduler = StepLR(optimizer , step_size= 200, gamma= 0.5)
    return [optimizer], [scheduler]
#------------------------------------------------------------------------------------
#to have leave-one-out cross validation: 
import csv
search_results = dict()
df_dict = {}
df = pd.read_csv('nirs with MRI outcome for CNN_30min.csv', header = None)
label = pd.read_csv('MRI outcome for CNN_30min.csv', header = None)
hie = pd.read_csv('HIE_fixed_for_MRI_30min.csv', header = None)
ids = df.iloc[:,0]
ids = list(dict.fromkeys(ids))
for i in ids:
    train_signal = df[df.iloc[:,0] != i]
    train_label = label[label.iloc[:,0] != i]
    train_signal = train_signal.values
    train_signal = train_signal[:,1:]
    # to standardize the signals 
    train_mean = np.mean(train_signal)
    train_std = np.std(train_signal)
    train_signal = (train_signal - train_mean)/train_std
    train_signal = train_signal.reshape(-1,1,300)
    train_hie = hie[hie.iloc[:,0] != i]
    train_hie = train_hie.values
    train_hie = train_hie[:,2]
    train_hie = train_hie.reshape(-1,1)
    train_label = train_label.values
    train_label = train_label[:,1]
    train_label = train_label.reshape(-1,1)
    # since the labels are imbalanced, we used pos-weight to consider this bias
    pos_weight = ((len(train_label)-sum(train_label))/sum(train_label))
    print(pos_weight)
    test_signal = df[df.iloc[:,0] == i]
    test_signal = test_signal.values
    test_signal = test_signal[:,1:]
    test_signal = (test_signal - train_mean)/train_std
    test_signal = test_signal.reshape(-1,1,300)
    test_label = label[label.iloc[:,0] == i]
    test_label = test_label.values
    test_label = test_label[:,1]
    test_label = test_label.reshape(-1,1)
    test_hie = hie[hie.iloc[:,0] == i]
    test_hie = test_hie.values
    test_hie = test_hie[:,2]
    test_hie = test_hie.reshape(-1,1)
    print(i)
    dataset = list(zip(train_signal, train_hie, train_label))
    test_dataset = list(zip(test_signal, test_hie,  test_label))
    # Load in dataloader
    train_loader = DataLoader(dataset, batch_size= 128, shuffle=True)
    test_loader = DataLoader(test_dataset)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight = (torch.tensor(pos_weight)).cuda())
    model = CNN()
    logger = pl.loggers.TensorBoardLogger('logs/', name='final')
    trainer = pl.Trainer(accelerator="gpu", devices= 1, max_epochs= 2000, logger=logger)
    trainer.fit(model, train_loader)
    trainer.test(model, test_loader)
    model.eval()
    test_signal = test_signal.reshape(-1,1,300)
    test_hie = test_hie.reshape(-1,1)
    y_hat = model(torch.tensor(test_signal).float(), torch.tensor(test_hie).float())
    y_hat = y_hat.reshape(-1)
    y_hat = pd.DataFrame(y_hat.detach().numpy())
    filename = f'{i}.csv'
    df_dict[filename] = filename
    y_hat.to_csv(filename, header=None, index=False)

#---------------------------------------------------------------------------------------------
