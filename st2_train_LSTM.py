# import packages
import torch
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# call preprocessing function
from st1_preprocessing_vars import preprocess_vars
df = preprocess_vars()

# set device and seed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

torch.manual_seed(42)
np.random.seed(42)

if torch.cuda.is_available():
  torch.cuda.manual_seed(42)
  torch.cuda.manual_seed_all(42)


# 1 --- DEFINE RiverData custom class
class RiverData(torch.utils.data.Dataset):

  # first function is the __init__() function, which loads data from input file
  def __init__(self, df, target, datecol, seq_len, pred_len):
    self.df = df
    self.datecol = datecol
    self.target = target
    self.seq_len = seq_len # length of single sequence of input data
    self.pred_len = pred_len # length of forward-forecasting sequence

  # second function sets the index to the date column
  def setIndex(self):
    self.df.set_index(self.datecol, inplace = True)

  # third function defines the length of the input dataset, minus the
  # training sequence length minus the prediction length
  def __len__(self):
    return len(self.df) - self.seq_len - self.pred_len

  # fourth function returns a typle of a feature and a label, which is used for
  # model training. for time series, the feature is the past values for
  # training and the label are the future values to be predicted
  def __getitem__(self, idx):

    # raises warning if not enough data at index idx
    if len(self.df) <= (idx + self.seq_len + self.pred_len):
      raise IndexError(
          f'Index {idx} out of bounds for dataset size {len(self.df)}')

    # pulls values for training sequence and assigns to df_piece
    df_piece = self.df[idx:idx + self.seq_len].values

    # converts df_piece into a tensor data type
    feature = torch.tensor(df_piece, dtype = torch.float32)

    # pulls the target data
    label_piece = self.df[self.target][
        idx + self.seq_len:
        idx + self.seq_len + self.pred_len].values

    # converts label_piece to tensor data type
    label = torch.tensor(label_piece, dtype = torch.float32)

    return (feature, label)
  

# 2 --- NORMALIZE DATAFRAME
raw_df = df.drop('datetime', axis = 1, inplace = False)
scaler = MinMaxScaler()

## apply transformations and make new df
df_scaled = scaler.fit_transform(raw_df)

df_scaled = pd.DataFrame(df_scaled, columns = raw_df.columns)
df_scaled['datetime'] = df['datetime']

## reassign
df = df_scaled


# 3 --- SET DATA SPLIT
train_size = int(0.7 * len(df))
test_size = int(0.2 * len(df))
val_size = len(df) - train_size - test_size

seq_len = 14
pred_len = 1
num_features = 3

## single asterix unpacks list
common_args = ['ssc_mg_L', 'datetime', seq_len, pred_len]

## pulled from beginning of df to train_size
train_dataset = RiverData(df[:train_size], *common_args)
train_dataset.setIndex()

# pulled from the end of train_size to the end of val_size
val_dataset = RiverData(df[train_size: train_size + val_size], *common_args)
val_dataset.setIndex()

## pulled from the end of train + val to the end of the df
test_dataset = RiverData(df[train_size + val_size: len(df)], *common_args)
test_dataset.setIndex()


# 4 --- SET HYPERPARAMETERS
## number of training examples used in one iteration to update model params
BATCH_SIZE = 492

SHUFFLE = False # order matters for time series
DATA_LOAD_WORKERS = 1

## learning rate determines step size at which params are updated while training
learning_rate = 0.007240183391544309

## adds penalty to loss function based on magnitude of model weights, preventing
## overreliance on single parameter
weight_decay = 0.000013647269057168989


# 5 -- DEPLOY DATALOADER
## double asterix unpacks dictionary
common_args = {'batch_size': BATCH_SIZE, 'shuffle': SHUFFLE}
train_loader = DataLoader(train_dataset, **common_args)
val_loader = DataLoader(val_dataset, **common_args)
test_loader = DataLoader(test_dataset, **common_args)


# 6 --- DEFINE PYTORCH LSTM MODEL
class BasicLSTMNetwork(torch.nn.Module):
  ## __init__() function sets up layers and defines model params
  def __init__(self, seq_len, pred_len):
    ## call base class constructor
    super().__init__()
    self.seq_len = seq_len
    self.pred_len = pred_len
    self.num_features = num_features
    self.n_layers = 1

    ## define size of hidden state
    self.n_hidden = 128

    ## define layers for combining across time series
    self.lstm1 = torch.nn.LSTM(input_size = self.num_features,
                               hidden_size = self.n_hidden,
                               num_layers = self.n_layers,
                               batch_first = True)
    self.relu = torch.nn.ReLU()
    self.fc1 = torch.nn.Linear(self.n_hidden * self.seq_len, self.pred_len)

  def init_hidden(self, batchsize):
    device = next(self.parameters()).device
    hidden_state = torch.zeros(self.n_layers,
                               batchsize,
                               self.n_hidden,
                               device = device)
    cell_state = torch.zeros(self.n_layers,
                             batchsize,
                             self.n_hidden,
                             device = device)
    return hidden_state, cell_state

  ## forward() function defines how forward pass computation operates
  ## gradients are stored inside FC layer objects
  ## each training example needs the old gradient erased
  def forward(self, x):
    batchsize, seqlen, featlen = x.size()
    self.hidden_states = self.init_hidden(batchsize)
    lstm_out, self.hidden_states = self.lstm1(x, self.hidden_states)
    lstm_out = lstm_out.contiguous().view(batchsize, -1)
    lstm_out = self.relu(lstm_out)
    lstm_out = self.fc1(lstm_out)
    return lstm_out
  

# 7 --- SET LOSS FUNCTION AND OPTIMIZER
model = BasicLSTMNetwork(seq_len, pred_len)
model = model.to(device)
loss = torch.nn.MSELoss().to(device)

# adam optimizer generally best option
optimizer = torch.optim.Adam(model.parameters(),
                             lr = learning_rate,
                             weight_decay = weight_decay)


# 8 --- MODEL PARAMETER CHECKS
## check one
for gen in model.parameters():
  print(gen.shape)

## check two
for i, (f, l) in enumerate(train_loader):
  print('features shape: ', f.shape)
  print('labels shape: ', l.shape)
  break


# 9 --- DEFINE EVALUATION METRICS
# define metrics
import numpy as np
import matplotlib.pyplot as plt
epsilon = np.finfo(float).eps

def wape_function(y, y_pred):
    # Weighted Average Percentage Error metric in the interval [0; 100]
    y = np.array(y)
    y_pred = np.array(y_pred)
    nominator = np.sum(np.abs(np.subtract(y, y_pred)))
    denominator = np.add(np.sum(np.abs(y)), epsilon)
    wape = np.divide(nominator, denominator) * 100.0
    return wape

def nse_function(y, y_pred):
    y = np.array(y)
    y_pred = np.array(y_pred)
    return (1-(np.sum((y_pred-y)**2)/np.sum((y-np.mean(y))**2)))


def evaluate_model(model, data_loader, plot=False):
    # following line disables dropout and batch normalization if they
    # are part of the model.

    model.eval()
    all_inputs = torch.empty((0, seq_len, num_features))
    all_labels = torch.empty(0, pred_len)
    for inputs, labels in data_loader:
        all_inputs = torch.vstack((all_inputs, inputs))
        all_labels = torch.vstack((all_labels, labels))

    with torch.no_grad():
        all_inputs = all_inputs.to(device)
        outputs = model(all_inputs).detach().cpu()
        avg_val_loss = loss(outputs, all_labels)
        nse = nse_function(all_labels.numpy(), outputs.numpy())
        wape = wape_function(all_labels.numpy(), outputs.numpy())

    print(f'NSE : {nse}', end=' ')
    print(f'WAPE : {wape}', end=' ')
    print(f'Validation Loss: {avg_val_loss}')
    model.train()

    if plot is True:
        plt.figure(figsize=(16, 6))
        plt.plot(np.array(all_labels.cpu()[:400]), 
                 color = '#836953', 
                 label = 'observations')
        plt.plot(np.array(outputs.cpu()[:400]), 
                 color = '#B19CD8', 
                 linestyle = '-', 
                 label = 'predictions')
        metrics_text = f"NSE: {nse:.3f}\nWAPE: {wape:.3f}"
        plt.text(0.28, 0.97, metrics_text, transform = plt.gca().transAxes,
        fontsize = 12, verticalalignment = 'top', bbox = dict(boxstyle = 'round', 
                                                            facecolor = 'wheat', 
                                                            alpha = 0.5))
        plt.legend()
        plt.title(f'LSTM with Optuna Optimization [Sequence Length: {seq_len}]')
        plt.show()

    return avg_val_loss


# 10 --- INITIATE MODEL TRAINING
num_epochs = 300
best_val_loss = float('inf')
patience = 10

for epoch in range(num_epochs):
  model.train()
  epoch_loss = []
  for batch_idx, (inputs, labels) in enumerate(train_loader):
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    loss_val = loss(outputs, labels)

    # calculate gradients for backpropagation
    loss_val.backward()
    # update weights based on gradients
    optimizer.step()
    # reset gradients --> avoid gradient accumulation
    optimizer.zero_grad()
    epoch_loss.append(loss_val.item())

    avg_train_loss = sum(epoch_loss) / len(epoch_loss)
    print(f'Epoch {epoch + 1}: Training Loss: {avg_train_loss}', end = ' ')
    avg_val_loss = evaluate_model(model, val_loader, device)

    # check for improvement
    if avg_val_loss < best_val_loss:
      best_val_loss = avg_val_loss
      epochs_no_improve = 0

      # save best model
      torch.save(model.state_dict(), 'best_model.pth')
    else:
      epochs_no_improve += 1
      if epochs_no_improve == patience:
        print('Early stopping!')
        # load next model before stopping
        model.load_state_dict(torch.load('best_model.pth'))
        break


# 11 --- VISUALIZE TEST DATA RESULT
evaluate_model(model, test_loader, plot = True)