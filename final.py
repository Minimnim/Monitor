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
#-------------------------------------------------------------------------------------------
#functions 
def making_test_val_ids(df, n):
    IDs = df[0].unique()
    random_ids = pd.Series(IDs).sample(n , replace= False)
    return random_ids
#ATTENTION: this function is different from previous ones, since it does not have validation
def making_train_test_val_signals(df, label, hie, n):
    a = making_test_val_ids(df, n)
    train_signal = df[~df[0].isin(a)]
    train_label = label[~label[0].isin(a)]
    test_IDs = a[:n]
    train_signal = train_signal.values
    train_signal = train_signal[:,1:]
    train_mean = np.mean(train_signal)
    train_std = np.std(train_signal)
    train_signal = (train_signal - train_mean)/train_std
    train_hie = hie[~hie[0].isin(a)]
    train_hie = train_hie.values
    train_hie = train_hie[:,3]
    train_label = train_label.values
    train_label = train_label[:,1]
    test_signal = df[df[0].isin(test_IDs)]
    test_signal = test_signal.values
    test_signal = test_signal[:,1:]
    test_signal = (test_signal - train_mean)/train_std
    test_label = label[label[0].isin(test_IDs)]
    test_label = test_label.values
    test_label = test_label[:,1]
    test_hie = hie[hie[0].isin(test_IDs)]
    test_hie = test_hie.values
    test_hie = test_hie[:,3]
    print(test_IDs)
    return train_signal, train_label, train_hie, test_signal, test_label, test_hie

def separate_epochs(train_signal, epoch_length):
    num_epochs = 5890 // epoch_length
    separated_epochs = np.split(train_signal, num_epochs, axis=1)
    return separated_epochs

def shuffle_epochs(epochs):
    np.random.shuffle(epochs)
    return epochs

def reconstruct_signals(epochs):
    reconstructed_signals = np.concatenate(epochs, axis=1)
    return reconstructed_signals

def separate_pieces(signal, num_pieces):
    piece_length = 5890 // num_pieces
    pieces = np.split(signal, num_pieces, axis=1)
    return pieces

def generate_new_signals(dataset, num_signals, num_rep, epoch_length, num):
    filtered_dataset = [(train_signal, train_hie, train_label) for train_signal, train_hie,  train_label in dataset if train_label == num and train_hie == num]
    filtered_signals = np.array([train_signal for train_signal, _ , _ in filtered_dataset])
    filtered_signals = filtered_signals.reshape(-1, 1, 5890)
    combined_signals = []
    for _ in range(num_rep):
        selected_indices = np.random.choice(filtered_signals.shape[0], size=num_signals, replace=False)
        selected_signals = filtered_signals[selected_indices]

        combined_epochs = []
        for train_signal in selected_signals:
            epochs = separate_epochs(train_signal, epoch_length)
            combined_epochs += epochs

        shuffled_epochs = shuffle_epochs(combined_epochs)
        reconstructed_signals = reconstruct_signals(shuffled_epochs)
        combined_signals.append(reconstructed_signals)

    combined_signals = np.array(combined_signals)
    # combined_signals = combined_signals.reshape(num_rep, 1, 5890)
    combined_signals = combined_signals.reshape(num_rep, -1)[ :, :5890]
    return combined_signals

def generate_new_signals_from_one(dataset, num_pieces):
    new_signals = []
    for signal, hie, label in dataset:
        signal = signal.reshape(1, 5890)
        pieces = separate_pieces(signal, num_pieces)
        permutations = list(itertools.permutations(pieces))
        for perm in permutations:
            # shuffled_pieces = shuffle_epochs(perm)
            new_signal = np.concatenate(perm, axis=1)
            new_signals.append((new_signal, hie, label))
    return new_signals
#-----------------------------------------------------------------------------------------------
class CNN(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.layer1 = nn.Sequential(
      nn.Conv1d(1, 8, kernel_size= 19, stride=5, padding=0),
      nn.BatchNorm1d(8),
      nn.LeakyReLU())
    self.layer2 = nn.Sequential(
      nn.Conv1d(8, 8, kernel_size= 11, stride=3, padding=0),
      nn.BatchNorm1d(8),
      nn.LeakyReLU())
    self.layer3 = nn.Sequential(
      nn.Conv1d(8, 16, kernel_size= 7, stride = 1, padding = 0),
      nn.BatchNorm1d(16),
      nn.LeakyReLU())
    self.layer4 = nn.Sequential(
      nn.Conv1d(16, 16, kernel_size = 5, stride = 1, padding = 0),
      nn.BatchNorm1d(16),
      nn.LeakyReLU())
    self.layer5_conv1 = nn.Sequential(
      nn.Conv1d(2, 1, kernel_size = 1, stride = 1, padding = 0))
    self.layer6 = nn.Sequential(nn.Dropout(p = 0.2))
    self.fc1 = nn.Linear(6064, 1)
  

  def forward(self, x, hie):
    out = self.layer1(x)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out_flatten = out.flatten(1)
    extended_hie = torch.repeat_interleave(hie, 6064)
    extended_hie = extended_hie.reshape(hie.shape[0], -1)
    out = torch.cat((out_flatten, extended_hie), dim=-1)
    out = out.reshape( hie.shape[0],2, 6064)
    out = self.layer5_conv1(out)
    out = self.layer6(out)
    out = self.fc1(out)
    return out


  def training_step(self, batch_loader, batch_idx):
    X, hie, Y = batch_loader
    Y_preds = self(X.float(), hie.float())
    Y_preds = Y_preds.reshape(-1,1)
    loss = loss_fn(Y_preds, Y.float())
    acc = accuracy(Y_preds, Y, task = "binary")
    self.log('train_loss', loss) 
    self.log('train_accuracy', acc)    
    # self.log('val auc', auc)
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
search_results = dict()
df = pd.read_csv('nirs with MRI outcome for CNN_10hours_overfree.csv', header = None)
label = pd.read_csv('MRI outcome for CNN_10hours_overfree.csv', header = None)
hie = pd.read_csv('HIE_fixed_for_MRI_overfree.csv', header = None)
ids = df.iloc[:,0]
ids = list(dict.fromkeys(ids))
for i in ids:
    train_signal = df[df.iloc[:,0] != i]
    train_label = label[label.iloc[:,0] != i]
    train_signal = train_signal.values
    train_signal = train_signal[:,1:]
    train_mean = np.mean(train_signal)
    train_std = np.std(train_signal)
    train_signal = (train_signal - train_mean)/train_std
    train_hie = hie[hie.iloc[:,0] != i]
    train_hie = train_hie.values
    train_hie = train_hie[:,3]
    train_label = train_label.values
    train_label = train_label[:,1]
    pos_weight = ((len(train_label)-sum(train_label))/sum(train_label))
    print(pos_weight)
    test_signal = df[df.iloc[:,0] == i]
    test_signal = test_signal.values
    test_signal = test_signal[:,1:]
    test_signal = (test_signal - train_mean)/train_std
    test_label = label[label.iloc[:,0] == i]
    test_label = test_label.values
    test_label = test_label[:,1]
    test_hie = hie[hie.iloc[:,0] == i]
    test_hie = test_hie.values
    test_hie = test_hie[:,3]
    print(i)
    dataset = list(zip(train_signal, train_hie, train_label))
    test_dataset = list(zip(test_signal, test_hie,  test_label))
    #augmentation for training dataset
    new_dataset = generate_new_signals_from_one(dataset, 5)
    new_signal = np.array([signal for signal, _,_ in new_dataset])
    new_signal = new_signal.reshape(len(new_dataset), -1)[ :, :5890]
    new_signal = pd.DataFrame(new_signal)
    new_signal = new_signal.values
    new_signal = new_signal.reshape(-1,1,5890)
    new_hie = np.array([hie for _, hie, _ in new_dataset])
    new_hie = pd.DataFrame(new_hie)
    new_hie = new_hie.values
    new_hie = new_hie.reshape(-1,1)
    new_label = np.array([label for _, _,label in new_dataset])
    new_label = pd.DataFrame(new_label)
    new_label = new_label.values
    new_label = new_label.reshape(-1,1)
    newer_dataset = list(zip(new_signal, new_hie, new_label))
    #augmentation for test dataset
    new_test = generate_new_signals_from_one(test_dataset, 5)
    new_signal_test = np.array([signal for signal, _,_ in new_test])
    new_signal_test = new_signal_test.reshape(len(new_test), -1)[ :, :5890]
    new_signal_test = pd.DataFrame(new_signal_test)
    new_signal_test = new_signal_test.values
    new_signal_test = new_signal_test.reshape(-1,1,5890)
    new_hie_test = np.array([hie for _, hie, _ in new_test])
    new_hie_test = pd.DataFrame(new_hie_test)
    new_hie_test = new_hie_test.values
    new_hie_test = new_hie_test.reshape(-1,1)
    new_label_test = np.array([label for _, _,label in new_test])
    new_label_test = pd.DataFrame(new_label_test)
    new_label_test = new_label_test.values
    new_label_test = new_label_test.reshape(-1,1)
    newer_test = list(zip(new_signal_test, new_hie_test, new_label_test))
    # Load in dataloader
    train_loader = DataLoader(newer_dataset, batch_size= 128, shuffle=True)
    test_loader = DataLoader(newer_test)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight = torch.tensor(pos_weight))
    model = CNN()
    logger = pl.loggers.TensorBoardLogger('logs/', name='final')
    trainer = pl.Trainer(accelerator="gpu", devices= 1, max_epochs= 2000, logger=logger)
    trainer.fit(model, train_loader)
    trainer.test(model, test_loader)
    model.eval()
    # Set the print options to display the entire tensor
    torch.set_printoptions(threshold=torch.inf)
    test_signal = test_signal.reshape(-1,1,5890)
    test_hie = test_hie.reshape(-1,1)
    y_hat = model(torch.tensor(test_signal).float(), torch.tensor(test_hie).float())
    print(y_hat)

#---------------------------------------------------------------------------------------------