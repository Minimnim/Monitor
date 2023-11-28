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
#make config files available
parser = argparse.ArgumentParser(description='Load a configuration file.')
parser.add_argument('config_file', type=str, nargs='?', default='conf.yaml', help='Path to the config file (default: default_config.yaml)')
args = parser.parse_args()
config_file = args.config_file
#---------------------------------------------------------------------------------------------
#functions 
def making_test_val_ids(df, n):
    IDs = df[0].unique()
    random_ids = pd.Series(IDs).sample(n , replace= False)
    return random_ids

def making_train_test_val_signals(df, label, hie, n):
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
    val_signal = df[df[0].isin(val_IDs)]
    val_signal = val_signal.values
    val_signal = val_signal[:,1:]
    val_signal = (val_signal - train_mean)/train_std
    val_label = label[label[0].isin(val_IDs)]
    val_label = val_label.values
    val_label = val_label[:,1]
    val_hie = hie[hie[0].isin(val_IDs)]
    val_hie = val_hie.values
    val_hie = val_hie[:,3]
    print(test_IDs)
    print(val_IDs)
    return train_signal, train_label, train_hie, test_signal, test_label, test_hie,  val_signal, val_label, val_hie

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
#----------------------------------------------------------------------------------------------
class FeatureExtractor(nn.Module):
    def __init__(self, in_channels,out_channels, kernel_size,pool_size):
        super(FeatureExtractor, self).__init__()
        self.blocks = nn.ModuleList()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_size = pool_size
      
        for idx in range(len(self.in_channels)):
            # Conv1d layer
            conv_layer = nn.Conv1d(in_channels= self.in_channels[idx], out_channels= self.out_channels[idx], kernel_size= self.kernel_size[idx])
            self.blocks.append(conv_layer)

            # MaxPool1d layer
            pool_layer = nn.MaxPool1d(self.pool_size)
            self.blocks.append(pool_layer)

            # LeakyReLU activation
            relu_layer = nn.LeakyReLU()
            self.blocks.append(relu_layer)

            # Update in_channels for the next block
            

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        return x

class CNN(pl.LightningModule):
  def __init__(self,lr ,in_channels,out_channels,StepLR_setp ,StepLR_gamma, kernel_size, signal_length, pool_size):
    super().__init__()
    self.StepLR_setp = StepLR_setp
    self.StepLR_gamma = StepLR_gamma
    self.lr = lr
    self.in_channels = in_channels 
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.signal_length = signal_length
    self.pool_size = pool_size

    self.feature_extractor = FeatureExtractor(in_channels = self.in_channels, out_channels=self.out_channels, kernel_size = self.kernel_size, pool_size = self.pool_size)
    sample_input = torch.randn(1, 1, self.signal_length)
    output_tensor = self.feature_extractor(sample_input)
    self.n_dim = output_tensor.size(0) * output_tensor.size(1) * output_tensor.size(2)
    self.fc_layer = nn.Linear(self.n_dim,1)
    self.layer6_conv1 = nn.Sequential(
     nn.Conv1d(2, 1, kernel_size = 1, stride = 1, padding = 0))
    self.drop_out = nn.Sequential(nn.Dropout(p = 0.2))

  def forward(self, x, hie):
    x = self.feature_extractor(x)
    out_flatten = x.flatten(1)
    extended_hie = torch.repeat_interleave(hie, self.n_dim)
    extended_hie = extended_hie.reshape(hie.shape[0], -1)
    out = torch.cat((out_flatten, extended_hie), dim=-1)
    out = out.reshape( hie.shape[0],2, self.n_dim)
    out = self.layer6_conv1(out)
    out = self.drop_out(out)
    out = self.fc_layer(out)
    return out

  def training_step(self, batch_loader, batch_idx):
    X, hie, Y = batch_loader
    Y_preds = self(X.float(), hie.float())
    Y_preds = Y_preds.reshape(Y.shape[0],1)
    loss = loss_fn(Y_preds, Y.float())
    acc = accuracy(Y_preds, Y, task = "binary")
    self.log('train_loss', loss) 
    self.log('train_accuracy', acc)    
    return {'loss': loss, 'accuracy':acc} 
    
  def validation_step(self, batch_loader, batch_idx):
    X, hie, Y = batch_loader
    Y_preds = self(X.float(), hie.float())
    Y_preds = Y_preds.reshape(Y.shape[0],1)
    loss = loss_fn(Y_preds, Y.float())
    acc = accuracy(Y_preds, Y, task = "binary")
    self.log('val_accuracy', acc)
    self.log('val_loss', loss)
    return {'loss': loss, 'accuracy':acc} 
  
  def test_step(self, batch_loader, batch_idx):
    X, hie, Y = batch_loader
    Y_preds = self(X.float(), hie.float())
    Y_preds = Y_preds.reshape(Y.shape[0],1)
    loss = loss_fn(Y_preds, Y.float())
    acc = accuracy(Y_preds, Y, task = "binary")
    self.log('test_accuracy', acc)
    self.log('test_loss', loss)
    return {'loss': loss, 'accuracy':acc} 
  
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    scheduler = StepLR(optimizer , step_size= self.StepLR_setp, gamma= self.StepLR_gamma)
    return [optimizer], [scheduler]
#------------------------------------------------------------------------------------
#to have k-fold cross validation: 
k_fold = 5
search_results = dict()
for i in range(k_fold):
    df = pd.read_csv('nirs with MRI outcome for CNN_10hours_overfree.csv', header = None)
    label = pd.read_csv('MRI outcome for CNN_10hours_overfree.csv', header = None)
    hie = pd.read_csv('HIE_fixed_for_MRI_overfree.csv', header = None)
    train_signal, train_label, train_hie, test_signal, test_label, test_hie, val_signal, val_label, val_hie = making_train_test_val_signals(df, label, hie, 8)
    train_signal = train_signal.reshape(-1,1,5890)
    train_hie = train_hie.reshape(-1,1)
    train_label = train_label.reshape(-1,1)
    test_signal = test_signal.reshape(-1,1,5890)
    test_hie = test_hie.reshape(-1,1)
    test_label = test_label.reshape(-1,1)
    val_signal = val_signal.reshape(-1,1,5890)
    val_hie = val_hie.reshape(-1,1)
    val_label = val_label.reshape(-1,1)
    dataset = list(zip(train_signal, train_hie, train_label))
    val_dataset = list(zip(val_signal, val_hie, val_label))
    test_dataset = list(zip(test_signal, test_hie,  test_label))
    train_loader = DataLoader(dataset, batch_size= 64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size= 8)
    test_loader = DataLoader(test_dataset)
    # Load Config File 
    with open(config_file, 'r') as file:
      data = yaml.safe_load(file)
    #--------------------------------------------------------------------------------------
    for idx in range(len(data["in_channels"])) :
            loss_fn = nn.BCEWithLogitsLoss(weight = torch.tensor(data["loss_weight"]).cuda())
            model = CNN(lr = data["lr"],StepLR_setp = data["StepLR_setp"] ,StepLR_gamma = data["StepLR_gamma"], in_channels=data["in_channels"][idx], out_channels=data["out_channels"][idx],kernel_size=data["kernel_size"][idx], signal_length = data["signal_length"], pool_size=data["pool_size"])
            logger = pl.loggers.TensorBoardLogger('logs/', name='simple_nn')
            trainer = pl.Trainer(accelerator="gpu", devices= 1, max_epochs= 3000, logger=logger)
            trainer.fit(model, train_loader, val_loader)
            trainer.test(model, test_loader)
            print(idx,  )
            model_name = f"model_{idx}_" + str(data["lr"]) + "_" + str(data["StepLR_setp"]) + "_"+ str(data["StepLR_gamma"])+ "_" +str(data["loss_weight"])
            #search_results[model_name] = logger.experiment.get_scalar('test_accuracy')
            checkpoint_path = f"model_CheckPoint/model_checkpoint_{model_name}.ckpt"
            trainer.save_checkpoint(checkpoint_path)
#------------------------------------------------------------------------------------
