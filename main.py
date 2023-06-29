import pandas as pd
import os as os
import torch
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings # supress warnings
from sklearn.preprocessing import MinMaxScaler

import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

warnings.filterwarnings('ignore')


# In[2]:


warnings.filterwarnings('ignore')
base_path = os.getcwd()
proj_path = base_path.replace("00_Code", "")
print(base_path)
print(proj_path)
os.chdir(proj_path)
# print("Path now set to: " + os.getcwd())


# In[3]:


torch.manual_seed(1234)
np.random.seed(1234)


# ## Load and prepare data

# In[4]:


df_books_data = pd.read_csv("0006_02_01_books_upsampled_MLbase.csv")
df_books_data.describe()

df_books_data.dtypes

df_books_data = df_books_data.drop(columns=['n_small',
                                            'n_big',
                                            'description'])

df_books_data['cat'] = df_books_data['cat'].astype('category')
df_books_data['subcat'] = df_books_data['subcat'].astype('category')
df_books_data['price'] = df_books_data['price'].str.replace("$", "")
df_books_data['price'] = df_books_data['price'].astype('float')


# In[5]:


df_books_data_model_NAs = df_books_data[df_books_data.isna().any(axis=1)]
df_books_data_model_NAs.head()


# In[6]:


print(len(df_books_data)) # 2989
df_books_data_model = df_books_data.dropna()
print(len(df_books_data_model)) #2076


# In[7]:


df_books_data_model['n_revs_sampled'] = np.where((df_books_data_model['n_freq'] >= df_books_data_model['n_unfreq']),
                                                 df_books_data_model['n_freq'],
                                                 df_books_data_model['n_unfreq'])

df_books_data_model = df_books_data_model.drop(columns=['j1', 'j2', 'j3', 'j4', 'j5',
                                                        'p1', 'p2', 'p3', 'p4', 'p5',
                                                         'is_j_to_p_upsampling',
                                                        'n_unfreq', 'n_freq'
                                                        #, 'asin'
                                                        ])

df_books_data_model = df_books_data_model.rename({'j1_sampled': 'j1',
                                                  'j2_sampled': 'j2',
                                                  'j3_sampled': 'j3',
                                                  'j4_sampled': 'j4',
                                                  'j5_sampled': 'j5',
                                                  'p1_sampled': 'p1',
                                                  'p2_sampled': 'p2',
                                                  'p3_sampled': 'p3',
                                                  'p4_sampled': 'p4',
                                                  'p5_sampled': 'p5',
                                                  'cat': 'categoriezz',
                                                  'subcat': 'subcategoriezz'
                                                  }, axis=1)  # new method


del df_books_data
df_books_data_model.head()



# ## Rescaling (Normalization)
# 

# In[8]:


df_books_data_model.head()
df_books_data_model_unscaled = df_books_data_model.copy()


# In[9]:


cols_to_normalize = [
    #    'j1', 'j2', 'j3', 'j4', 'j5',
    #    'p1', 'p2', 'p3', 'p4', 'p5',
    #    'polarity_j',
    #    'polarity_p',
    #   'posimbalance_j',
    'price', 'n_revs_sampled'
]

scaler = MinMaxScaler()
df_books_data_model[cols_to_normalize] = scaler.fit_transform(df_books_data_model[cols_to_normalize])


# In[10]:


df_books_data_model.head()


# ## Add polarity and imbalance values

# In[11]:


# from Schoenmueller et al. (2021)
# polarity scores
df_books_data_model['polarity_j'] = df_books_data_model['j1'] + df_books_data_model['j5']

bins = np.linspace(0,1,30,endpoint=True)
plt.figure(figsize=(2,2)) #change your figure size as per your desire here
plt.hist(df_books_data_model[['polarity_j']], bins=bins)
plt.show()

df_books_data_model['polarity_p'] = df_books_data_model['p1'] + df_books_data_model['p5']

bins = np.linspace(0,1,30,endpoint=True)
plt.figure(figsize=(2,2)) #change your figure size as per your desire here
plt.hist(df_books_data_model[['polarity_p']], bins=bins)
plt.show()



# In[12]:


#drop low polarity scores
#df_books_data_model = df_books_data_model.drop(df_books_data_model[(df_books_data_model['polarity_j'] <= 0.4)].index)
#df_books_data_model = df_books_data_model.drop(df_books_data_model[(df_books_data_model['polarity_p'] > 0.65)].index)

bins = np.linspace(0,1,30,endpoint=True)
plt.figure(figsize=(2,2)) #change your figure size as per your desire here
plt.hist(df_books_data_model[['polarity_j']], bins=bins)
plt.show()

bins = np.linspace(0,1,30,endpoint=True)
plt.figure(figsize=(2,2)) #change your figure size as per your desire here
plt.hist(df_books_data_model[['polarity_p']], bins=bins)
plt.show()


# In[13]:


# positive imbalance scores
df_books_data_model['posimbalance_j'] = (df_books_data_model['j4'] + df_books_data_model['j5']) / (df_books_data_model['j1'] + df_books_data_model['j2'] + df_books_data_model['j4'] + df_books_data_model['j5'])

bins = np.linspace(0,1,30,endpoint=True)
plt.figure(figsize=(2,2)) #change your figure size as per your desire here
plt.hist(df_books_data_model[['posimbalance_j']], bins=bins)
plt.show()


# In[14]:


#drop low positive imbalance scores
#df_books_data_model = df_books_data_model.drop(df_books_data_model[(df_books_data_model['posimbalance_j'] <= 0.50)].index)

bins = np.linspace(0,1,30,endpoint=True)
plt.figure(figsize=(2,2)) #change your figure size as per your desire here
plt.hist(df_books_data_model[['posimbalance_j']], bins=bins)
plt.show()


# ## Rearrange DF

# In[15]:


df_books_data_model_copy = df_books_data_model
#df_books_data_model = df_books_data_model.drop(['asin'], axis=1)
df_books_data_model['init_index'] = df_books_data_model.index.values

df_books_data_model = df_books_data_model[['init_index',
                                           'p1', 'p2', 'p3', 'p4', 'p5',
                                           'j1', 'j2', 'j3', 'j4', 'j5',
                                           'polarity_j',
                                           'polarity_p',
                                           #'posimbalance_j',
                                           'price', 'n_revs_sampled', 'categoriezz', 'subcategoriezz']]


# In[16]:


df_books_data_model.head()


# ## Categories cleaning & dummies

# In[17]:


# What categories do we have?
subcatz = df_books_data_model.subcategoriezz.unique().tolist()
catz = df_books_data_model.categoriezz.unique().tolist()
print(len(subcatz))
len_subcatz = len(subcatz)

print(len(catz))
len_catz = len(catz)


# In[18]:


sorted(subcatz)


# In[19]:


sorted(catz)


# In[20]:


# Get dummies

# Cats conatin spaces- replace
df_books_data_model['categoriezz'] = df_books_data_model['categoriezz'].str.replace(' ', '_')
df_books_data_model['subcategoriezz'] = df_books_data_model['subcategoriezz'].str.replace(' ', '_')

df_books_data_model.head()


# In[21]:


df_books_data_model = pd.get_dummies(df_books_data_model, columns=['categoriezz', 'subcategoriezz'], prefix = ['cat', 'subcat'])
#df_books_data_model = df_books_data_model.drop(columns=['categoriezz', 'subcategoriezz'])
df_books_data_model.head()


# In[22]:


print(len(df_books_data_model))
df_books_data_model = df_books_data_model.dropna()
print(len(df_books_data_model))


# In[23]:


print(len(df_books_data_model.iloc[:, 15:].columns)) # dummies worked (128)


# In[24]:


n_data = len(df_books_data_model)

# Creating a dataframe with 80% values of original dataframe
train_set_df = df_books_data_model.sample(frac = 0.80, random_state = 1234)

# Creating dataframe with rest of values
remainig_set_df = df_books_data_model.drop(train_set_df.index)

# splitting training and validation
val_set_df = remainig_set_df.sample(frac = 0.5, random_state = 1234)
test_set_df = remainig_set_df.drop(val_set_df.index)

print(len(train_set_df))
print(len(val_set_df))
print(len(test_set_df))


# In[25]:


print(len(val_set_df.columns))


# In[26]:


val_set_df.head()


# In[27]:


print(len(val_set_df.iloc[:, 6:].columns))


# In[28]:


pol_j = val_set_df.iloc[:, 11]
pol_p = val_set_df.iloc[:, 12]

dens_df = pd.concat([pol_j, pol_p], axis=1)
dens_df.plot(kind='density') # or pd.Series()


# In[29]:


columns_to_select = list(range(6, 12)) + list(range(13, len(val_set_df.columns)))
val_set_df.iloc[:, columns_to_select]


# In[ ]:





# In[30]:


# define dataset with dataloader
class ReviewDataset(Dataset):

    def __init__(self, data):
        # Initialize data, download, etc.
        # read with numpy or pandas
        xy = data.to_numpy()
        self.n_samples = xy.shape[0]

        # class label, features
        self.y_data = torch.from_numpy(xy[:, 1:6].astype(np.float32)) # size [n_samples, 1]
        self.x_data = torch.from_numpy(xy[:, columns_to_select].astype(np.float32))
        #self.x_polarity_p = torch.from_numpy(xy[:, 12].astype(np.float32)) # size [n_samples, n_features]
        self.init_index = torch.from_numpy(xy[:, 0].astype(np.float32)) # size [n_samples, n_features]

        # npdata = data.to_numpy()
        # self.n_samples = len(npdata)
        #
        # # class label, features
        # self.y_data = torch.tensor(npdata.iloc[:, [1,2,3,4,5]].astype(np.float32))
        # self.x_data = torch.tensor(npdata.iloc[:, 6:].astype(np.float32))
        # self.x_polarity_p = torch.tensor(npdata.loc[:, ["polarity_p"]].astype(np.float32))
        # self.init_index = torch.tensor(npdata.loc[:, ["init_index"]].astype(np.float32))

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], \
               self.y_data[index], \
               self.init_index[index]
               #self.x_polarity_p[index], \

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

# create dataset
train_set = ReviewDataset(train_set_df)
val_set = ReviewDataset(val_set_df)
test_set = ReviewDataset(test_set_df)


# In[31]:


# Define the batch size
batch_size = 32


# In[32]:


train_loader = DataLoader(dataset = train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset = val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset = test_set, batch_size=batch_size, shuffle=False)


# In[33]:


input_size = sum([5, # 5 js
                  1, # polarity j
                  #1, # polarity p
                  1, # price
                  1, # n_revs_sampled
                  len_catz, # len categories
                  len_subcatz # len subcategories
                 ]
                 )

print(input_size)
hidden_size = [256, 128, 64, 32]
num_classes = 5


# In[34]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

torch.autograd.set_detect_anomaly(True)


class NN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.fc4 = nn.Linear(hidden_size[2], hidden_size[3])
        self.fc5 = nn.Linear(hidden_size[3], num_classes)
        self.relu = nn.ReLU()

        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc5(x)

        return x


class ConstraintLayer(nn.Module):
    def __init__(self):
        super(ConstraintLayer, self).__init__()

    def forward(self, output, x_data):

        # Apply constraint

        zeros_mask_j1 = x_data[:, 0] == 0
        zeros_mask_j2 = x_data[:, 1] == 0

        output_corrected = output.clone()  # Create a copy
        output_corrected = F.softmax(output_corrected, dim=1)

        output_corrected = output_corrected.clone()
        output_corrected[:, 0].masked_fill_(zeros_mask_j1, 0)
        output_corrected[:, 1].masked_fill_(zeros_mask_j2, 0)

        ## NORMALIZE
        # Calculate the sum of each row
        row_sums = torch.sum(output_corrected, dim=1)

        # Normalize each row by dividing by the row sums
        output_corrected = output_corrected / row_sums.unsqueeze(1)

        #output_corrected[:, 0][zeros_mask_j1] = 0
        #output_corrected[:, 1][zeros_mask_j2] = 0

        return output_corrected

NNmodel = NN(input_size, hidden_size, num_classes)
constraint_layer = ConstraintLayer()


# In[35]:


# Hyperparameters
num_epochs = 100 # fine tuned based on 300 first
total_samples = n_data
n_iterations = math.ceil(total_samples / batch_size)

# Define a loss function and optimizer
criterion = nn.KLDivLoss()
#criterion = nn.CrossEntropyLoss()
#criterion = torch.nn.NLLLoss()


#scheduler = StepLR(optimizer, step_size=10, gamma=1)


# In[36]:


learning_rate = 0.0001 # 0.00001 looks promising
optimizer = torch.optim.Adam(NNmodel.parameters(), lr=learning_rate)


# In[38]:


torch.autograd.set_detect_anomaly(True)

min_valid_loss = np.inf
NNmodel_performance_df = pd.DataFrame(columns = ['epoch', 'training_loss', 'validation_loss'])

for epoch in range(num_epochs):

    train_loss = 0.0
    NNmodel.train()     # Optional when not using Model Specific layer

    for x_data, \
        y_data, \
        init_index in train_loader:

        optimizer.zero_grad()

        outputs = NNmodel(x_data)
        outputs_before_constraint = outputs.clone()  # Model output before applying constraint

        outputs_corrected = constraint_layer(outputs, x_data)
        outputs_corrected_log_softmax = F.log_softmax(outputs_corrected)

        # STUFF FOR COMPLEX LOSS FCT
        #outputs_KLDiv = torch.nn.functional.log_softmax(outputs, dim=1)
        outputs_KLDiv = outputs_corrected_log_softmax
#        outputs_polarity_p = torch.nn.functional.softmax(outputs, dim=1)

        # Add the first and fifth value of each row of the input tensor
 #       polarity_p_pred = outputs_polarity_p[:, 0] + outputs_polarity_p[:, 4]

        # Calculate loss
        loss = criterion(outputs_KLDiv, y_data)
        #loss = criterion(outputs, y_data)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    valid_loss = 0.0
    NNmodel.eval()     # Optional when not using Model Specific layer

    for x_data, \
        y_data, \
        init_index in val_loader:

        outputs = NNmodel(x_data)
        outputs_before_constraint = outputs.clone()  # Model output before applying constraint

        outputs_corrected = constraint_layer(outputs, x_data)
        outputs_corrected_log_softmax = F.log_softmax(outputs_corrected)

        # STUFF FOR COMPLEX LOSS FCT
        #outputs_KLDiv = torch.nn.functional.log_softmax(outputs, dim=1)
        outputs_KLDiv = outputs_corrected_log_softmax
 #       outputs_polarity_p = torch.nn.functional.softmax(outputs, dim=1)

        # Add the first and fifth value of each row of the input tensor
#        polarity_p_pred = outputs_polarity_p[:, 0] + outputs_polarity_p[:, 4]

        # Calculate loss
        loss = criterion(outputs_KLDiv, y_data)
        #loss = criterion(outputs, y_data)

        valid_loss = loss.item() * x_data.size(0)

    print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_loader)} \t\t Validation Loss: {valid_loss / len(val_loader)}')

    # SAVE EPOCH LOSS DATA
    NNmodel_performance_df.loc[epoch, 'epoch'] = epoch
    NNmodel_performance_df.loc[epoch, 'training_loss'] = train_loss / len(train_loader)
    NNmodel_performance_df.loc[epoch, 'validation_loss'] = valid_loss / len(val_loader)

    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss

        #Saving State Dict
        torch.save(NNmodel.state_dict(), '0006_03_00_saved_NNmodel.pth')


# In[39]:


from matplotlib.pylab import plt
from scipy.interpolate import splrep, splev

# Get them data
train_values = NNmodel_performance_df.training_loss.astype('float')
val_values = NNmodel_performance_df.validation_loss.astype('float')
epochs = NNmodel_performance_df.epoch.astype('float')

# Smooth it
smooth_train = splrep(epochs, train_values, s=0.0002)
smooth_train_values = splev(epochs, smooth_train)

smooth_val = splrep(epochs, val_values, s=0.0002)
smooth_val_values = splev(epochs, smooth_val)

df_smooth_val = pd.DataFrame(smooth_val_values)
#df_smooth_val = pd.concat([df_smooth_val, epochs])
df_smooth_val.columns = ["val"]

# get min of smoothed
min_index = df_smooth_val['val'].idxmin()
min_index




# In[40]:


# Plot it
# Create a figure with 2 subplots, one for each variable
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
fig.subplots_adjust(hspace=4, top=2)
fig.suptitle('Training and validation loss', fontsize=30)

axs[0].plot(epochs, train_values, label='Training Loss', color='deepskyblue')
axs[0].plot(epochs, val_values, label='Validation Loss', color='darkorange')
axs[0].set_title('Training and Validation Loss: Unsmoothed')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].legend(loc='best')
#axs[0].axvline(min_index, color='red', linestyle='--')  # Add vertical red line



axs[1].plot(epochs,smooth_train_values, label = 'Training Loss Smoothed', color='deepskyblue')
axs[1].plot(epochs,smooth_val_values, label = 'Validation Loss Smoothed', color='darkorange')
axs[1].set_title('Training and Validation Loss: Smoothed')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Loss')
axs[1].legend(loc='best')
axs[1].axvline(min_index, color='darkorange', linestyle='--', linewidth=0.8)  # Add vertical red line
axs[1].text(min_index+5, 0.015, "$min_{val}$ ="+ f'{min_index}', color='darkorange', ha='left', va='bottom')  # Add red text label

# axs[1].get_ylim()[1]
fig.tight_layout()
plt.show()


# In[41]:


# Plot it
# Create a figure with 2 subplots, one for each variable

plt.plot(epochs, train_values, label='Training Loss', color='deepskyblue')
plt.plot(epochs, val_values, label='Validation Loss', color='darkorange')
plt.title('Training and Validation Loss:' f' {num_epochs} ' 'epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='best')
#axs[0].axvline(min_index, color='red', linestyle='--')  # Add vertical red line

# axs[1].get_ylim()[1]
fig.tight_layout()
plt.show()


# In[42]:


#Get raw data
y_data_raw_val = val_set_df[['p1', 'p2', 'p3', 'p4', 'p5']]
x_data_raw_val = val_set_df.drop(columns=['p1', 'p2', 'p3', 'p4', 'p5'])
#
# # Get column names of original data for later
columns_y_data_raw_val = y_data_raw_val.columns.tolist()
columns_x_data_raw_val = x_data_raw_val.columns.tolist()
#
# # Get initial data for prediction
# x_data = df_books_data_model.drop(columns=['p1', 'p2', 'p3', 'p4', 'p5']).values
# y_data = df_books_data_model[['p1', 'p2', 'p3', 'p4', 'p5']].values
#
# # tensor setup
# x_data_tensor = test_set.x_data
# y_data_tensor = test_set.y_data
# indexes = test_set.init_index

# Predictions
with torch.no_grad():
    y_pred_precorrection = NNmodel(val_set.x_data)
    y_pred = constraint_layer(y_pred_precorrection, val_set.x_data)

# Convert predictions to numpy array
y_pred_val = y_pred.numpy()
# Save the input data and the predictions in a CSV file
# Combine the input data and the predictions into a single numpy array
output_data_val = np.concatenate((y_pred_val, y_data_raw_val, x_data_raw_val), axis=1)

# Create a pandas dataframe from the numpy array
output_df_val = pd.DataFrame(output_data_val, columns=['p1_pred', 'p2_pred', 'p3_pred', 'p4_pred',
                                                         'p5_pred'] + columns_y_data_raw_val + columns_x_data_raw_val)

output_df_val['polarity_p_pred'] = output_df_val['p1_pred'] + output_df_val['p5_pred']

# Save the dataframe to a CSV file
#output_df_val.to_csv('02_Output/0006_03_01_p_prediction_model7_MAbased_update_val.csv', index=False)
output_df_val.head()


# In[43]:


bins = np.linspace(0,1,30,endpoint=True)

# Create a figure with 3 subplots arranged in one row
fig, axs = plt.subplots(3, 1, figsize=(5, 5), sharex = 'all')

# Create a histogram for each dataset and add it to a subplot
axs[0].hist(output_df_val[['polarity_j']], bins=bins)
axs[1].hist(output_df_val[['polarity_p']], bins=bins)
axs[2].hist(output_df_val[['polarity_p_pred']], bins=bins)

# Set the title for each subplot
axs[0].set_title('polarity_j')
axs[1].set_title('polarity_p')
axs[2].set_title('polarity_p_pred')

# Show the plot
plt.show()


# In[44]:


#plt, ax = plt.subplots()

plt.scatter(x  = output_df_val.polarity_p, y = output_df_val.polarity_p_pred)
plt.axline((0, 0), slope=1, color='r', linestyle='dotted')
plt.xlabel("polarity_p")
plt.ylabel("polarity_p_pred")
plt.text(0.6, 1, 'slope=1 reference', fontsize=10, color = "red", verticalalignment='top')

plt.show()


# In[45]:


df = output_df_val

df[['p1', 'p2', 'p3', 'p4', 'p5', 'p1_pred', 'p2_pred', 'p3_pred', 'p4_pred', 'p5_pred']] = df[['p1', 'p2', 'p3', 'p4', 'p5', 'p1_pred', 'p2_pred', 'p3_pred', 'p4_pred', 'p5_pred']].astype('float64')

# Create a figure with 5 subplots, one for each variable
fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(20,7))
#fig.subplots_adjust(hspace=8)
fig.suptitle('Validation set: more centered distribution, actual vs. predicted, regression excl. constant', fontsize=30)

for i, var in enumerate(['1', '2', '3', '4', '5']):
    # Calculate the regression slope and R-squared without intercept using Statsmodels
    x = df[f'p{var}']
    y = df[f'p{var}_pred']
    #X = sm.add_constant(x)
    OLSmodel = sm.OLS(y, x).fit()
    slope = OLSmodel.params[0]
    regression_line = OLSmodel.params[0] * x
    #regression_line = OLSmodel.params[0] + OLSmodel.params[1] * x
    r2 = OLSmodel.rsquared
    print(f'r2 =  {r2}')
    axs[i].scatter(x, y, alpha = 0.3, s = 12, color = "#022F55")
    axs[i].set_xlabel("Actual", fontsize = 15)
    axs[i].set_ylabel("Predicted", fontsize = 15)

    print(OLSmodel.summary())


    # Add the slope and R-squared as text in the upper right corner of the subplot
    textstr = f'Slope = {slope:.2f}\nR-squared = {r2:.2f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.4)
    axs[i].text(0.04, 0.95, textstr, transform=axs[i].transAxes, fontsize=15, verticalalignment='top', bbox=props)
    axs[i].set_title(f'{var}' + "-star", fontsize=20)
    axs[i].plot(x, regression_line, color='red', alpha = 1, linestyle = "dashed", linewidth = 1)

# Show the plot
plt.tight_layout(rect=[0, 0, 1, 0.95])


plt.show()


# In[ ]:





# In[46]:


import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Input data
x = np.array([1, 2, 3, 4, 5])  # Independent variable
y = np.array([3, 5, 7, 9, 11])  # Dependent variable

# Create and fit OLS model without intercept
model = sm.OLS(y, x)
results = model.fit()

# Regression line
regression_line = results.params[0] * x

# Scatter plot
plt.scatter(x, y, label='Data')

# Regression line plot
plt.plot(x, regression_line, color='red', label='Regression Line')

# Labels and legend
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# Show the plot
plt.show()


# # RESIDUALS

# In[47]:


import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

df = output_df_val

df[['p1', 'p2', 'p3', 'p4', 'p5', 'p1_pred', 'p2_pred', 'p3_pred', 'p4_pred', 'p5_pred']] = df[['p1', 'p2', 'p3', 'p4', 'p5', 'p1_pred', 'p2_pred', 'p3_pred', 'p4_pred', 'p5_pred']].astype('float64')

# Create a figure with 6 subplots (5 for residuals + 1 for total residuals)
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 12))
fig.suptitle('Validation Set: Residuals Analysis', fontsize=30)

# Create lists to store total residuals and subplot axes
total_residuals = []
subplot_axes = []

# Iterate over each variable
for i, var in enumerate(['1', '2', '3', '4', '5']):
    # Calculate the regression slope and R-squared without intercept using Statsmodels
    x = df[f'p{var}']
    y = df[f'p{var}_pred']
    OLSmodel = sm.OLS(y, x, hasconst=False).fit()
    slope = OLSmodel.params[f'p{var}']
    r2 = OLSmodel.rsquared
    residuals = y - slope * x

    # Store total residuals for later use
    total_residuals.append(residuals)

    # Scatter plot of residuals
    ax = axs[i//3, i%3]
    ax.scatter(x, residuals, alpha=0.3, s=12, color="#022F55")
    ax.axhline(0.00, color='red', linestyle='--')  # Add red horizontal line at 0.00
    ax.set_xlabel("Actual", fontsize=15)
    ax.set_ylabel("Residual", fontsize=15)
    ax.set_title(f'{var}-star Residuals', fontsize=20)
    subplot_axes.append(ax)

    # Calculate measure(s) of total residual (e.g., mean absolute error)
    total_residual_mean = np.mean(residuals)
    total_residual_mae = np.mean(np.abs(residuals))

    # Add the measure(s) as text in the upper right corner of the subplot
    textstr = f'Mean = {total_residual_mean:.2f}\nMAE = {total_residual_mae:.2f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.4)
    ax.text(0.04, 0.95, textstr, transform=ax.transAxes, fontsize=15, verticalalignment='top', bbox=props)

# Scatter plot of total residuals
total_residuals = np.concatenate(total_residuals)
ax = axs[1, 2]
ax.scatter(np.arange(len(total_residuals)), total_residuals, alpha=0.3, s=12, color="#022F55")
ax.axhline(0.00, color='red', linestyle='--')  # Add red horizontal line at 0.00
ax.set_xlabel("Index", fontsize=15)
ax.set_ylabel("Total Residual", fontsize=15)
ax.set_title("Total Residuals", fontsize=20)

# Calculate measure(s) of total residuals (e.g., mean absolute error)
total_residual_mean = np.mean(total_residuals)
total_residual_mae = np.mean(np.abs(total_residuals))

# Add measure(s) as text in the upper right corner of the subplot
textstr = f'Mean = {total_residual_mean:.2f}\nMAE = {total_residual_mae:.2f}'
props = dict(boxstyle='round', facecolor='white', alpha=0.4)
ax.text(0.04, 0.95, textstr, transform=ax.transAxes, fontsize=15, verticalalignment='top', bbox=props)

# Adjust the layout of subplots
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Show the plot
plt.show()


# # TESTING

# In[48]:


# Test the model and save the results
NNmodel.eval()

# Test the model
with torch.no_grad():
    total_loss = 0.0
    KL_div_loss = 0.0
    polarity_loss = 0.0
    n_samples = 0
    avg_total_loss = 0.0
    avg_KL_div_loss = 0.0
    avg_polarity_loss = 0.0

# x_polarity_p,

    for x_data, y_data, init_index in test_loader:
        # Forward pass
        outputs = NNmodel(x_data)
        outputs_before_constraint = outputs.clone()  # Model output before applying constraint

        outputs_corrected = constraint_layer(outputs, x_data)
        outputs_corrected_log_softmax = F.log_softmax(outputs_corrected)

        outputs_polarity_p = torch.nn.functional.softmax(outputs, dim=1)
        polarity_p_pred = outputs_polarity_p[:, 0] + outputs_polarity_p[:, 4]

        total_loss += criterion(outputs_corrected_log_softmax, y_data)

        # Update number of samples
        n_samples += y_data.size(0)

    # Calculate average loss
    avg_total_loss = total_loss / n_samples
    avg_KL_div_loss = KL_div_loss / n_samples
    avg_polarity_loss = polarity_loss / n_samples


    # Print losses avg
    print(f'Loss data on testing set with {n_samples:d} samples')
    print(f'AVG total loss: {avg_total_loss:.4f}')
    # Print losses abs
    print(f'ABS total loss: {total_loss.item():.4f}')



# In[49]:


n_samples


# In[ ]:





# In[50]:


#Get raw data
y_data_raw_test = test_set_df[['p1', 'p2', 'p3', 'p4', 'p5']]
x_data_raw_test = test_set_df.drop(columns=['p1', 'p2', 'p3', 'p4', 'p5'])
#
# # Get column names of original data for later
columns_y_data_raw_test = y_data_raw_test.columns.tolist()
columns_x_data_raw_test = x_data_raw_test.columns.tolist()
#
# # Get initial data for prediction
# x_data = df_books_data_model.drop(columns=['p1', 'p2', 'p3', 'p4', 'p5']).values
# y_data = df_books_data_model[['p1', 'p2', 'p3', 'p4', 'p5']].values
#
# # tensor setup
# x_data_tensor = test_set.x_data
# y_data_tensor = test_set.y_data
# indexes = test_set.init_index

# Predictions
with torch.no_grad():
    y_pred_precorrection = NNmodel(test_set.x_data)
    outputs_before_constraint = y_pred_precorrection.clone()  # Model output before applying constraint
    y_pred = constraint_layer(outputs_before_constraint, test_set.x_data)

    #outputs_corrected_log_softmax = F.log_softmax(outputs_corrected)
    #y_pred = torch.nn.functional.softmax(y_pred, dim=1)

# Convert predictions to numpy array
y_pred_test = y_pred.numpy()


# In[52]:


# Save the input data and the predictions in a CSV file
# Combine the input data and the predictions into a single numpy array
output_data_test = np.concatenate((y_pred_test, y_data_raw_test, x_data_raw_test), axis=1)

# Create a pandas dataframe from the numpy array
output_df_test = pd.DataFrame(output_data_test, columns=['p1_pred', 'p2_pred', 'p3_pred', 'p4_pred', 'p5_pred'] + columns_y_data_raw_test + columns_x_data_raw_test)

output_df_test['polarity_p_pred'] = output_df_test['p1_pred'] + output_df_test['p5_pred']

# Save the dataframe to a CSV file
output_df_test.to_csv('0006_03_02_ML_model_outputs_testingdata_UPDATED.csv', index=False)


# In[53]:


output_df_test.head()


# In[54]:


len(output_df_test)


# In[55]:


df = output_df_test

df[['p1', 'p2', 'p3', 'p4', 'p5', 'p1_pred', 'p2_pred', 'p3_pred', 'p4_pred', 'p5_pred']] = df[['p1', 'p2', 'p3', 'p4', 'p5', 'p1_pred', 'p2_pred', 'p3_pred', 'p4_pred', 'p5_pred']].astype('float64')

# Create a figure with 5 subplots, one for each variable
fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(20,7))
#fig.subplots_adjust(hspace=8)
fig.suptitle('TESTING set: more centered distribution, actual vs. predicted, regression excl. constant', fontsize=30)

for i, var in enumerate(['1', '2', '3', '4', '5']):
    # Calculate the regression slope and R-squared without intercept using Statsmodels
    x = df[f'p{var}']
    y = df[f'p{var}_pred']
    OLSmodel = sm.OLS(y, x, hasconst=False).fit()
    slope = OLSmodel.params[f'p{var}']
    r2 = OLSmodel.rsquared
    print(f'r2 =  {r2}')
    axs[i].scatter(x, y, alpha = 0.3, s = 12, color = "#022F55")
    axs[i].set_xlabel("Actual", fontsize = 15)
    axs[i].set_ylabel("Predicted", fontsize = 15)

    # Add the slope and R-squared as text in the upper right corner of the subplot
    textstr = f'Slope = {slope:.2f}\nR-squared = {r2:.2f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.4)
    axs[i].text(0.04, 0.95, textstr, transform=axs[i].transAxes, fontsize=15, verticalalignment='top', bbox=props)
    axs[i].set_title(f'{var}' + "-star", fontsize=20)
    axs[i].plot(x, slope * x, color='red', alpha = 1, linestyle = "dashed", linewidth = 1)

# Show the plot
plt.tight_layout(rect=[0, 0, 1, 0.95])


plt.show()


# In[ ]:




