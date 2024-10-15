import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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


def create_sequences(data, sequence_length):
    xs= []
    for i in range(len(data) - sequence_length+1):
        x = data[i:i+sequence_length]
        xs.append(x)
    return np.array(xs)



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * divisor)
        pe[:, 1::2] = torch.cos(position * divisor)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class Transformer(nn.Module):
    def __init__(self, features, d_model, head, dropout, num_layers):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(features, d_model)
        self.encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=head, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(d_model, 256)
        self.fc2 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        X = self.embedding(X)  # 60 x 64 x 512 
        X = self.encoding(X)
        X = self.transformer_encoder(X)
        X = self.fc1(X[-1, :, :])
        X = self.dropout(X)
        X = self.fc2(X)
        return X

def predict(stock,sequence_length):
    number_of_prediction=2
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    df = get_selected_indicator(stock)
    # df=pd.read_csv(os.path.join(settings.BASE_DIR,'myapp',f'l{stock}.csv'))
    df = df.tail(sequence_length+number_of_prediction-1)
    print(f'\nlast date in inference {df['date'].iloc[-1]} and closeing is{df['close'].iloc[-1]}\n')
    
    # Drop the date column (assumed irrelevant for prediction)
    df = df.drop(columns=['date'])
    
    # Get the number of columns dynamically
    columns = len(df.columns)
    feature = columns  # number of features
    
    # Scale the data
    scaler =StandardScaler()
    scaled_data = scaler.fit_transform(df)
    # create sequence 
    X_test=create_sequences(scaled_data,sequence_length)
    print(f'shape{X_test.shape}')
    # Convert the data to a PyTorch tensor
    X_test = torch.tensor(X_test, dtype=torch.float32)
    
    
    
    # Initialize the Transformer model
    model = Transformer(feature, d_model, head, dropout, num_layers)
    model.to(device)
    
    # Load the pre-trained model
    model_path = os.path.join(settings.BASE_DIR, 'pth', f'{stock}.pth')
    model.load_state_dict(torch.load(model_path))
    
    # Set the model to evaluation mode
    model.eval()
    print(X_test.shape)
    # Make predictions without gradient tracking
    with torch.no_grad():
        X_test = X_test.transpose(0, 1).to(device)
        test_output = model(X_test)
    df = df.tail(sequence_length*number_of_prediction)
    print(f'test outout {test_output}')
    # Convert prediction to CPU and detach from the computation graph
    test_output_cpu = test_output.cpu().numpy()
    print(test_output_cpu.shape)
    shaped_output=np.concatenate((test_output_cpu,np.zeros((test_output_cpu.shape[0],scaled_data.shape[1]-1))),axis=1)
    
    # Perform inverse transformation
    result = scaler.inverse_transform(shaped_output)[:,0]
    print(f'rescaled result is {result}')
    # Extract the first element (single rescaled value)
    return result
