import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, n_input, lstm_size, num_classes=4, keep_prob=0.5):
        super(LSTM, self).__init__()
        self.lstm_size = lstm_size
        self.keep_prob = keep_prob
        
        self.lstm = nn.LSTM(input_size=n_input, hidden_size=lstm_size, batch_first=True)
        self.dropout = nn.Dropout(1 - keep_prob)
        
        self.fc1 = nn.Linear(lstm_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        c_last = c_n[-1] 
        
        x = self.fc1(c_last)
        x = self.bn1(x)
        x = F.softplus(x)
        x = self.dropout(x)
        
        prediction = self.fc2(x)
        return prediction
