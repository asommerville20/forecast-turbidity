# import packages
import torch
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import optuna


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
from st2_train_LSTM import RiverData
  

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
from st2_train_LSTM import BasicLSTMNetwork, __init__, init_hidden, forward
  

# 7 --- SET LOSS FUNCTION
loss = torch.nn.MSELoss().to(device)


# 8 --- DEFINE EVALUATION METRICS
# define metrics
epsilon = np.finfo(float).eps
from st2_train_LSTM import wape_function, nse_function, evaluate_model


# 9 -- INITIATE OPTUNA CALIBRATION
def objective(trial):
    # define the search space of the hyper-parameters -- optuna uses
    # bayesian optimization to find the optimal values of the hyperparameters.
    learning_rate = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)

    model = BasicLSTMNetwork(seq_len, pred_len)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr = learning_rate,
                                 weight_decay=weight_decay)

    # shoot for 50 next
    num_epochs = 300

    best_val_loss = float('inf')

    # keep this between 5-10
    patience = 10

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = []
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss_val = loss(outputs, labels)

            # calculate gradients for back propagation
            loss_val.backward()

            # update the weights based on the gradients
            optimizer.step()

            # reset the gradients, avoid gradient accumulation
            optimizer.zero_grad()
            epoch_loss.append(loss_val.item())

        avg_train_loss = sum(epoch_loss)/len(epoch_loss)
        print(f'Epoch {epoch+1}: Training Loss: {avg_train_loss}', end=' ')
        avg_val_loss = evaluate_model(model, val_loader)

        # check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # save the best model
            torch.save(model.state_dict(), 'best_model_trial.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print('Early stopping!')
                # load the best model before stopping
                model.load_state_dict(torch.load('best_model_trial.pth'))
                break

        # report intermediate objective value
        trial.report(best_val_loss, epoch)

        # handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_loss

study = optuna.create_study(direction='minimize')

study.optimize(objective, n_trials = 200)

print('Number of finished trials:', len(study.trials))
print('Best trial:')
trial = study.best_trial

print('  Value (Best Validation Loss):', trial.value)
print('  Params:')
for key, value in trial.params.items():
    print(f'    {key}: {value}')


# 10 --- WRITE OPTIMIZATION HISTORY
import optuna.visualization as vis

# optimization history
fig1 = vis.plot_optimization_history(study)
fig1.write_html("optimization_history_lstm.html")