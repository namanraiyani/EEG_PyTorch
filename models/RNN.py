import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, n_input, rnn_size, num_classes=4, keep_prob=0.5):
        super(RNN, self).__init__()
        self.rnn_size = rnn_size
        self.keep_prob = keep_prob
        
        self.rnn = nn.RNN(input_size=n_input, hidden_size=rnn_size, batch_first=True)
        self.dropout = nn.Dropout(1 - keep_prob)
        
        self.fc1 = nn.Linear(rnn_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        out, h_n = self.rnn(x)
        h_last = h_n[-1]
        
        x = self.fc1(h_last)
        x = self.bn1(x)
        x = F.softplus(x)
        x = self.dropout(x)
        
        prediction = self.fc2(x)
        return prediction
