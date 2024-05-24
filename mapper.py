import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FC(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0, activation='relu', bias=True):
        super(FC, self).__init__()
        self.dropout = dropout
        self.fc = nn.Linear(in_dim, out_dim)
        if activation == 'relu':
            self.relu = nn.ReLU(inplace=True)
        elif activation == 'tanh':
            self.relu = nn.Tanh(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.fc(x)
        if self.relu:
            x = self.relu(x)
        if self.dropout > 0:
            x = nn.Dropout(self.dropout)(x)
        return x
    
class MLP(nn.Module):
    def __init__(self, dim_size, dropout=0.0, activation='relu', bias=True):
        super(MLP, self).__init__()
        layers = []
        for i in range(1, len(dim_size)):
            layers.append(FC(dim_size[i-1], dim_size[i], dropout, activation, bias))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        return self.ln(x)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, dim_self, dim_ref=None, num_heads=1, dropout=0.0, bias=True):
        super(MultiHeadAttention, self).__init__()
        self.dim_self = dim_self
        self.num_heads = num_heads
        self.head_dim = dim_self // num_heads
        if dim_ref is None:
            dim_ref = dim_self

        # Create linear layers for Q, K, and V
        self.fc_q = nn.Linear(dim_self, dim_self, bias=bias)
        self.fc_k = nn.Linear(dim_ref, dim_self, bias=bias)
        self.fc_v = nn.Linear(dim_ref, dim_self, bias=bias)
        
        # Create final linear layer
        self.fc_o = nn.Linear(dim_self, dim_self, bias=bias)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        n_batch = q.size(0)

        # Transform Q, K, V
        q = self.fc_q(q).view(n_batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.fc_k(k).view(n_batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.fc_v(v).view(n_batch, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores and output
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        
        # Reshape output
        output = output.transpose(1, 2).contiguous().view(n_batch, -1, self.dim_self)
        
        return self.fc_o(output)

class TransformerLayer(nn.Module):
    def __init__(self, dim_self, dim_ref, num_heads, mlp_dim, dropout=0.0, activation='relu', bias=False):
        super(TransformerLayer, self).__init__()
        self.ln1 = LayerNorm(dim_self)
        self.ln2 = LayerNorm(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, dropout, bias)
        self.mlp = MLP([dim_self, mlp_dim, dim_self], dropout, activation)


    def forward(self, x, y=None, mask=None):
        if y is None:
            y = x
        x = x + self.attn(self.ln1(x), self.ln1(y), self.ln1(y), mask)
        x = x + self.mlp(self.ln2(x))
        return x
    
    def forward_with_attn(self, x, y=None, mask=None):
        if y is None:
            y = x
        attn, x = self.attn(self.ln1(x), self.ln1(y), self.ln1(y), mask)
        x = x + self.mlp(self.ln2(x))
        return attn, x
    
class Transformer(nn.Module):
    def __init__(self, dim_self, dim_ref, num_heads, mlp_dim, num_layers, dropout=0.0, activation='relu', enc_dec=False):
        super(Transformer, self).__init__()
        self.dim_ref = dim_ref if dim_ref else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec: #cross attention
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_dim, dropout, activation))
            elif i % 2 == 1 and enc_dec: #self attention
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_dim, dropout, activation))
            else: #self or cross attention
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_dim, dropout, activation))
        self.transformer = nn.Sequential(*layers)
    
    def forward(self, x, y=None, mask=None):
        if y is None:
            y = x
        for i, layer in enumerate(self.transformer):
            if i % 2 == 0 and self.enc_dec:
                x = layer(x, y, mask)
            elif i % 2 == 1 and self.enc_dec:
                x = layer(x, x, mask)
            else:
                x = layer(x, y, mask)
        return x

    def forward_with_attn(self, x, y=None, mask=None):
        if y is None:
            y = x
        attns = []
        for i, layer in enumerate(self.transformer):
            if i % 2 == 0 and self.enc_dec:
                attn, x = layer.forward_with_attn(x, y, mask)
            elif i % 2 == 1 and self.enc_dec:
                attn, x = layer.forward_with_attn(x, x, mask)
            else:
                attn, x = layer.forward_with_attn(x, y, mask)
            attns.append(attn)
        return attns, x

# Map CLIP embeddings to Image prefix with fixed length
class PrefixMapper(nn.Module):
    def __init__(self, dim_clip, dim_embd, prefix_len, clip_len, num_layers):
        super(PrefixMapper, self).__init__()
        self.clip_len = clip_len
        self.fc = nn.Linear(dim_clip, clip_len*dim_embd)
        self.transformer = Transformer(dim_embd, None, num_heads=8, mlp_dim=2048, num_layers=num_layers)
        self.prefix_const = nn.Parameter(torch.rand(prefix_len, dim_embd), requires_grad=True)

    def forward(self, x):
        x = self.fc(x).view(x.size(0), self.clip_len, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_len:]
        return out

class AdditiveAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AdditiveAttention, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.context_vector = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x_proj = torch.tanh(self.input_proj(x))
        attn_weights = torch.softmax(self.context_vector(x_proj), dim=1)
        weighted_sum = torch.sum(attn_weights * x, dim=1)
        return weighted_sum, attn_weights

class KGMapperWithAttention(nn.Module):
    def __init__(self, dim_kg, dim_embd, kg_len, num_layers):
        super(KGMapperWithAttention, self).__init__()
        self.kg_len = kg_len
        self.fc = nn.Linear(dim_kg, dim_embd * kg_len)
        
        # Add an attention mechanism
        self.attention = AdditiveAttention(dim_embd * kg_len, dim_embd * kg_len)

        # Define MLP
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(dim_embd * kg_len, dim_embd * kg_len))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc(x).view(x.size(0), self.kg_len, -1)
        x, _ = self.attention(x)
        out = self.mlp(x)
        return out

#Map Info retrieved from KG with variable length to fixed length embedding using Attention Mechanism
class KGMapper(nn.Module):
    def __init__(self, dim_kg, dim_embd, kg_len, num_layers):
        super(KGMapper, self).__init__()
        self.kg_len = kg_len
        self.fc = nn.Linear(dim_kg, dim_embd*kg_len)
        self.transformer = Transformer(dim_embd, None, num_heads=8, mlp_dim=2048, num_layers=num_layers)
        self.kg_const = nn.Parameter(torch.rand(1, dim_embd), requires_grad=True)

    def forward(self, x):
        x = self.fc(x).view(x.size(0), self.kg_len, -1)
        out = self.transformer(x)
        return out


class KGMapperMLP(nn.Module):
    def __init__(self, dim_kg, dim_embd, kg_len, hidden_dim):
        super().__init__()
        self.kg_len = kg_len
        # Input projection layer
        self.fc1 = nn.Linear(dim_kg, hidden_dim)
        # Additional hidden layers
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, dim_embd * kg_len)
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(dim_embd * kg_len)

    def forward(self, x):
        # Apply the first fully connected layer and a ReLU activation
        x = F.relu(self.bn1(self.fc1(x)))
        # Apply the second fully connected layer and a ReLU activation
        x = F.relu(self.bn2(self.fc2(x)))
        # Apply the third fully connected layer
        x = self.bn3(self.fc3(x))
        # Reshape the output to fit the knowledge graph dimensions
        x = x.view(x.size(0), self.kg_len, -1)
        return x

if __name__ == '__main__':
    # Test PrefixMapper
    model = PrefixMapper(512, 128, 10, 10, 6)
    x = torch.rand(32, 512)
    y = model(x)
    print(y.shape)
    # Test KGMapper
    model = KGMapperWithAttention(512, 128, 10, 6)
    x = torch.rand(32, 512)
    y = model(x)
    print(y.shape)