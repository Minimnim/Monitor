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
#from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
import itertools
#from sklearn.utils.class_weight import compute_class_weight

#---------------------------------------------------------------------------------------------

def making_test_val_ids(df, n):
    IDs = df[0].unique()
    random_ids = pd.Series(IDs).sample(n , replace= False)
    return random_ids

def making_train_test_val_signals(df, label, n):
    a = making_test_val_ids(df, n)
    train_signal = df[~df[0].isin(a)]
    train_label = label[~label[0].isin(a)]
    test_IDs = a[:(n//2)]
    val_IDs = a[(n//2):]
    train_signal = train_signal.values
    train_signal = train_signal[:,1:]
    train_mean = np.mean(train_signal)
    train_std = np.std(train_signal)
    train_signal = (train_signal - train_mean)/train_std
    train_label = train_label.values
    train_label = train_label[:,1]
    test_signal = df[df[0].isin(test_IDs)]
    test_signal = test_signal.values
    test_signal = test_signal[:,1:]
    test_signal = (test_signal - train_mean)/train_std
    test_label = label[label[0].isin(test_IDs)]
    test_label = test_label.values
    test_label = test_label[:,1]
    val_signal = df[df[0].isin(val_IDs)]
    val_signal = val_signal.values
    val_signal = val_signal[:,1:]
    val_signal = (val_signal - train_mean)/train_std
    val_label = label[label[0].isin(val_IDs)]
    val_label = val_label.values
    val_label = val_label[:,1]
    print(test_IDs)
    print(val_IDs)
    return train_signal, train_label,  test_signal, test_label,  val_signal, val_label

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
    for signal, label in dataset:
        signal = signal.reshape(1, 5890)
        pieces = separate_pieces(signal, num_pieces)
        permutations = list(itertools.permutations(pieces))
        for perm in permutations:
            # shuffled_pieces = shuffle_epochs(perm)
            new_signal = np.concatenate(perm, axis=1)
            new_signals.append((new_signal, label))
    return new_signals


def generate_new_rows_amin(row, num_new_rows, window_percentage):
    new_rows = []
    for _ in range(num_new_rows):
        row_length = len(row)
        window_size = int(row_length * window_percentage)

        # Randomly select start indices for the windows
        start_idx_orig = random.randint(1, row_length - window_size)
        start_idx_swap = random.randint(1, row_length - window_size)

        # Generate the new row by swapping the windows
        new_row = row[:]
        new_row[start_idx_orig:start_idx_orig + window_size], new_row[start_idx_swap:start_idx_swap + window_size] = \
            new_row[start_idx_swap:start_idx_swap + window_size], new_row[start_idx_orig:start_idx_orig + window_size]

        new_rows.append(new_row)
    return new_rows

# def sigmoid(x):
#     return 1 / (1 + torch.exp(-x))

#-----------------------------------------------------------------------------------------------


class CNN(pl.LightningModule):
  def __init__(self):
    super().__init__()
    
    self.layer1 = nn.Sequential(
      nn.Conv1d(1, 16, kernel_size= 11, stride=2, padding=0),
      nn.BatchNorm1d(16),
      nn.LeakyReLU())
    
    self.layer2 = nn.Sequential(
      nn.Conv1d(16, 32, kernel_size= 7, stride=2, padding=0),
      nn.BatchNorm1d(32),
      nn.LeakyReLU())
    
    self.layer3 = nn.Sequential(
      nn.Conv1d(32, 32, kernel_size= 7, stride = 2, padding = 0),
      nn.BatchNorm1d(32),
      nn.LeakyReLU())
  
    self.layer6 = nn.Sequential(nn.Dropout(p = 0.2))
    
    self.n_dim = 23392
    self.fc1 = nn.Linear(self.n_dim, 1)
    
  def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer6(out)
    out_flatten = out.flatten(1)
    out = self.fc1(out_flatten)
    return out


  def training_step(self, batch_loader, batch_idx):
    X, Y = batch_loader
    Y_preds = self(X.float())
    # Y_preds = Y_preds.reshape(Y.shape[0],1)
    loss = loss_fn(Y_preds, Y.float())
    acc = accuracy(Y_preds, Y, task = "binary")
    self.log('train_loss', loss) 
    self.log('train_accuracy', acc)    
    # self.log('val auc', auc)
    return {'loss': loss, 'accuracy':acc} 
    
  def validation_step(self, batch_loader, batch_idx):
    X, Y = batch_loader
    Y_preds = self(X.float())
    # Y_preds = Y_preds.reshape(Y.shape[0],1)
    loss = loss_fn(Y_preds, Y.float())
    acc = accuracy(Y_preds, Y, task = "binary")
    self.log('val_accuracy', acc)
    self.log('val_loss', loss)
    return {'loss': loss, 'accuracy':acc} 
  
  def test_step(self, batch_loader, batch_idx):
    X, Y = batch_loader
    Y_preds = self(X.float())
    # Y_preds = Y_preds.reshape(Y.shape[0],1)
    loss = loss_fn(Y_preds, Y.float())
    acc = accuracy(Y_preds, Y, task = "binary")
    self.log('test_accuracy', acc)
    self.log('test_loss', loss)
    return {'loss': loss, 'accuracy':acc} 
  
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-5, weight_decay=1e-5)
    return optimizer
  
loss_fn = nn.BCEWithLogitsLoss()

#-----------------------------------------------------------------------------------------
#k_fold 
for i in range(10):
    #train the model with HIE labels
    df = pd.read_csv('nirs with MRI outcome for CNN_10hours_overfree.csv', header = None)
    label = pd.read_csv('HIE_fixed_for_MRI_overfree.csv', header = None)
    df = df.values
    df = df[:,1:]
    label = label.values
    label = label[:,1]
    dataset = list(zip(df, label))
    new_dataset = generate_new_signals_from_one(dataset, 5)
    new_signal = np.array([signal for signal, _ in new_dataset])
    new_signal = new_signal.reshape(len(new_dataset), -1)[ :, :5890]
    new_signal = pd.DataFrame(new_signal)
    new_signal = new_signal.values
    new_signal = new_signal.reshape(-1,1,5890)
    new_label = np.array([label for _ ,label in new_dataset])
    new_label = pd.DataFrame(new_label)
    new_label = new_label.values
    weight = (len(new_label) - sum(new_label))/ sum(new_label)
    new_label = new_label.reshape(-1,1)
    newer_dataset = list(zip(new_signal, new_label))
    train_loader = DataLoader(newer_dataset, batch_size=64, shuffle=True)
    model = CNN()
    loss_fn = nn.BCEWithLogitsLoss(weight = torch.tensor(weight).cuda())
    logger = pl.loggers.TensorBoardLogger('logs/', name='simple_nn')
    trainer = pl.Trainer(accelerator="gpu", devices= 1, max_epochs= 3000, logger=logger, 
                     default_root_dir="model")
    trainer.fit(model, train_loader)
    #freezing layers
    for name, param in model.named_parameters():
      if "fc1" in name or "layer6" in name:
        param.requires_grad = True
      else: 
        param.requires_grad = False
    
    #test on MRI labels
    df = pd.read_csv('nirs with MRI outcome for CNN_10hours_overfree.csv', header = None)
    label = pd.read_csv('MRI outcome for CNN_10hours_overfree.csv', header = None)
    train_signal, train_label, test_signal, test_label, val_signal, val_label = making_train_test_val_signals(df, label, 8)
    val_label = val_label.reshape(-1,1)
    val_signal = val_signal.reshape(-1,1,5890)
    test_label = test_label.reshape(-1,1)
    test_signal = test_signal.reshape(-1,1,5890)
    train_label = train_label.reshape(-1,1)
    train_signal = train_signal.reshape(-1,1,5890)
    dataset = list(zip(train_signal, train_label))
    val_dataset = list(zip(val_signal, val_label))
    test_dataset = list(zip(test_signal,  test_label))
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size= 64)
    test_loader = DataLoader(test_dataset)
    trainer = pl.Trainer( accelerator="gpu", devices= 1, max_epochs= 3000, logger=logger, 
                     default_root_dir="model")
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)