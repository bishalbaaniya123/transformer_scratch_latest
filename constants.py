import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = "data/all_data"
path = os.path.join(os.getcwd(), path).replace(os.sep, '/')

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 5

# Training the Model
teacher_forcing_ratio = 0.8

# Number of neurons in hidden layer
hidden_size = 512
