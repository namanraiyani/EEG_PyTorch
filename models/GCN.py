import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse
import numpy as np
from .lib_for_GCN import graph

class ChebyshevConv(nn.Module):
    def __init__(self, in_features, out_features, K, L):
        super(ChebyshevConv, self).__init__()
        self.K = K
        self.in_features = in_features
        self.out_features = out_features
        
        L = scipy.sparse.csr_matrix(L)
        L = graph.rescale_L(L, lmax=2)
        L = L.tocoo()
        indices = torch.from_numpy(np.vstack((L.row, L.col)).astype(np.int64))
        values = torch.from_numpy(L.data.astype(np.float32))
        shape = torch.Size(L.shape)
        self.register_buffer('L_indices', indices)
        self.register_buffer('L_values', values)
        self.L_shape = shape
        
        self.weight = nn.Parameter(torch.Tensor(K * in_features, out_features))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.Tensor(out_features))
        nn.init.zeros_(self.bias)

    def forward(self, x):
        N, M, Fin = x.shape
        x0 = x.permute(1, 2, 0).contiguous().view(M, -1)
        L = torch.sparse_coo_tensor(self.L_indices, self.L_values, self.L_shape)
        Xt = torch.zeros(self.K, M, Fin * N, device=x.device)
        Xt[0] = x0
        if self.K > 1:
            Xt[1] = torch.sparse.mm(L, x0)
        for k in range(2, self.K):
            Xt[k] = 2 * torch.sparse.mm(L, Xt[k-1]) - Xt[k-2]
        Xt = Xt.view(self.K, M, Fin, N).permute(3, 1, 2, 0).contiguous()
        Xt = Xt.view(N * M, Fin * self.K)
        out = torch.mm(Xt, self.weight) + self.bias
        out = out.view(N, M, self.out_features)
        return out

class GCN(nn.Module):
    def __init__(self, L, F, K, p, M, num_classes=4, dropout=0.5):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.p = p
        
        in_features = 1
        for i in range(len(F)):
            self.layers.append(ChebyshevConv(in_features, F[i], K[i], L[i]))
            in_features = F[i]
            
        self.fc_layers = nn.ModuleList()
        
        if len(L) > len(F):
            last_M = L[-1].shape[0]
        else:
            last_M = L[-1].shape[0]
            if p[-1] > 1:
                last_M //= p[-1]
            
        fc_in = last_M * F[-1]
        
        for i in range(len(M)):
            self.fc_layers.append(nn.Linear(fc_in, M[i]))
            fc_in = M[i]
            
        self.fc_out = nn.Linear(fc_in, num_classes)

    def forward(self, x):
        x = x.unsqueeze(2)
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = F.relu(x)
            
            if self.p[i] > 1:
                x = x.permute(0, 2, 1)
                x = F.max_pool1d(x, kernel_size=self.p[i], stride=self.p[i])
                x = x.permute(0, 2, 1)
                
        x = x.contiguous().view(x.size(0), -1)
        
        for layer in self.fc_layers:
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        prediction = self.fc_out(x)
        return prediction
