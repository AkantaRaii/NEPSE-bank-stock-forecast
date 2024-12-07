import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .transformer import Transformer
import os
from django.conf import settings 

from ..data_preprocessing.select_indicator import get_selected_indicator

# Set hyperparameters (can also be dynamic based on data)
head = 8           # number of heads in multihead attention
d_model = 512      # dimension of input for the encoder
dropout = 0.3      # dropout rate to deactivate neurons
num_layers = 6     # number of encoder layers
lr = 0.0001        # learning rate
# PositionalEncoding, Transformer, and predict function


def create_sequences_lasso(data, sequence_length):
    xs= []
    for i in range(len(data) - sequence_length+1):
        x = data[i:i+sequence_length]
        xs.append(x)
    return np.array(xs)
def create_sequences_pca(data,sequence_length):
        xs = []
        for i in range(len(data) - sequence_length+1):
            x = data[i:i + sequence_length]
            xs.append(x)
        return np.array(xs), 
class Predict:

    def predict_lasso(self,stock,sequence_length):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        df = get_selected_indicator(stock,'lasso')
        # df=pd.read_csv(os.path.join(settings.BASE_DIR,'myapp',f'l{stock}.csv'))
        df = df.tail(sequence_length)
        date=df['date']
        # Drop the date column (assumed irrelevant for prediction)
        df = df.drop(columns=['date'])

        # Get the number of columns dynamically
        columns = len(df.columns)
        feature = columns  # number of features

        # Scale the data 
        scaler =StandardScaler()
        scaled_data = scaler.fit_transform(df)
        print(f'scaled_data{scaled_data}')
        # create sequence 
        X_test=create_sequences_lasso(scaled_data,sequence_length)
        print(f'shape{X_test.shape}')
        # Convert the data to a PyTorch tensor
        X_test = torch.tensor(X_test, dtype=torch.float32)



        # Initialize the Transformer model
        model = Transformer(feature, d_model, head, dropout, num_layers)
        model.to(device)

        # Load the pre-trained model
        model_path = os.path.join(settings.BASE_DIR,'../','best_model','bestmodel_lasso', f'{stock}{sequence_length}.pth')
        model.load_state_dict(torch.load(model_path))

        # Set the model to evaluation mode
        model.eval()
        print(X_test.shape)
        # Make predictions without gradient tracking
        with torch.no_grad():
            X_test = X_test.transpose(0, 1).to(device)
            test_output = model(X_test)
        print(f'test outout {test_output}')
        # Convert prediction to CPU and detach from the computation graph
        test_output_cpu = test_output.cpu().numpy()
        print(test_output_cpu.shape)
        shaped_output=np.concatenate((test_output_cpu,np.zeros((test_output_cpu.shape[0],scaled_data.shape[1]-1))),axis=1)

        # Perform inverse transformation
        result = scaler.inverse_transform(shaped_output)[:,0]
        print(f'rescaled result is {result}')
        print(type(result))
        print(f"dates:{date.values}")
        # Extract the first element (single rescaled value)
        return result,date.values


    def predict_pca(self,stock,sequence_length):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        scaler =StandardScaler()

        data = get_selected_indicator(stock,'pca')
        data=data.tail(sequence_length)
        date=data['date']
        close=data['close'].values.reshape(-1,1)
        scaled_data=data.drop(columns=['date','close'])
        close = scaler.fit_transform(close)
        scaled_data['close_scaled']=np.squeeze(close)
        # Get the number of columns dynamically
        columns = len(scaled_data.columns)
        feature = columns  # number of features
        print(scaled_data)
        print(scaled_data.values)
        print(type(close))

        # Scale the data
        # create sequence 
        X_test=create_sequences_pca(scaled_data.values,sequence_length)
        print(X_test)
        
        # Convert the data to a PyTorch tensor
        X_test = torch.tensor(X_test, dtype=torch.float32)
        X_test = X_test.squeeze(1)  # Removes the second dimension
        print(X_test.shape)  # Should output: torch.Size([1, 9, 6])

        print(X_test)


        # Initialize the Transformer model
        model = Transformer(feature, d_model, head, dropout, num_layers)
        model.to(device)

        # Load the pre-trained model
        model_path = os.path.join(settings.BASE_DIR,'../','best_model','bestmodel_pca', f'{stock}{sequence_length}.pth')
        model.load_state_dict(torch.load(model_path))

        # Set the model to evaluation mode
        model.eval()
        print(X_test.shape)
        # Make predictions without gradient tracking
        with torch.no_grad():
            X_test = X_test.transpose(0, 1).to(device)
            test_output = model(X_test)
        print(f'test outout {test_output}')
        # Convert prediction to CPU and detach from the computation graph
        test_output_cpu = test_output.cpu().numpy()
        print(test_output_cpu.shape)
        shaped_output=np.concatenate((test_output_cpu,np.zeros((test_output_cpu.shape[0],scaled_data.shape[1]-1))),axis=1)

        # Perform inverse transformation
        result = scaler.inverse_transform(shaped_output)[:,0]
        print(f'rescaled result is {result}')
        print(type(result))
        # Extract the first element (single rescaled value)
        return result,date.values
        
