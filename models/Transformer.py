import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att_dim = embed_dim * num_heads
        self.input_proj = nn.Linear(embed_dim, self.att_dim)
        self.att = nn.MultiheadAttention(self.att_dim, num_heads, batch_first=True)
        self.output_proj = nn.Linear(self.att_dim, embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, inputs):
        x = self.input_proj(inputs)
        attn_output, _ = self.att(x, x, x)
        attn_output = self.output_proj(attn_output)
        
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out = self.layernorm2(out1 + ffn_output)
        return out

class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, maxlen, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.pos_emb = nn.Embedding(maxlen, embed_dim)
        self.maxlen = maxlen
        self.embed_dim = embed_dim

    def forward(self, x):
        positions = torch.arange(0, self.maxlen, device=x.device)
        positions = self.pos_emb(positions)
        x = x.view(-1, self.maxlen, self.embed_dim)
        out = x + positions
        return out

class Transformer(nn.Module):
    def __init__(self, maxlen=3, embed_dim=97, num_heads=8, ff_dim=64, num_classes=1):
        super(Transformer, self).__init__()
        self.embedding_layer = TokenAndPositionEmbedding(maxlen, embed_dim)
        self.transformer_block_1 = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.transformer_block_2 = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(embed_dim, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.transformer_block_1(x)
        x = self.transformer_block_2(x)
        
        x = x.permute(0, 2, 1)
        x = F.max_pool1d(x, kernel_size=x.shape[2]).squeeze(2)
        
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
