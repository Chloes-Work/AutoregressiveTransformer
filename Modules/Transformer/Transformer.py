import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import numpy as np

from Modules.Loss.MSELoss import MSELoss

class Transformer(nn.Module):
    def __init__(self, output_dim=512, embed_dim=512, n_blocks = 6, n_heads=4, ff_dim=2048, dropout=0.1, norm = None, n_mels=80):
        super().__init__()
        self.encoder = TransformerEncoder(output_dim, embed_dim, n_blocks, n_heads, ff_dim, dropout, norm, n_mels)
        self.decoder = TransformerDecoder(output_dim, embed_dim, n_blocks, n_heads, ff_dim, dropout, norm, n_mels)
        
        #may define outputdim
        #self.linear_out = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        memory = x
        x = self.encoder(x)
        x = self.decoder(x, memory)
        return x


class AutoregressiveTransformer(nn.Module):
    def __init__(self, output_dim=512, embed_dim=512, n_blocks = 6, n_heads=4, ff_dim=2048, dropout=0.1, norm = None, n_mels=80):
        super().__init__()
        self.encoder = TransformerEncoder(output_dim, embed_dim, n_blocks, n_heads, ff_dim, dropout, norm, n_mels)
        self.decoder = TransformerDecoder(output_dim, embed_dim, n_blocks, n_heads, ff_dim, dropout, norm, n_mels)
        
        #may define outputdim
        #self.linear_out = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        x_en = self.encoder(x)
        x = self.decoder(x_en)
        
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, output_dim=512, embed_dim=512, n_blocks = 6, n_heads=4, ff_dim=2048, dropout=0.1, norm=None, n_mels=80):
        super().__init__()
        self.linear_in = nn.Linear(n_mels, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.encoders = torch.nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim,
                n_heads,
                ff_dim,
                dropout,
                norm
            ) for _ in range(n_blocks)
        ])
        self.linear_out = nn.Linear(embed_dim, output_dim)

    def forward(self,x):
        x = self.linear_in(x)
        x = self.positional_encoding(x)
        for layer in self.encoders:
            x = layer(x)
        x = self.linear_out(x)
        return x

    def plot_attention_maps(self, label, x):
        x = self.positional_encoding(x)
        attention_maps = []
        for layer in self.encoders:
            _, attn_map = layer.attn(x, mask=None, return_attention=True)
            attention_maps.append(attn_map)
            layer.plot_attention_maps(label, [attn_map])
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, output_dim=512, embed_dim=512, n_blocks = 6, n_heads=4, ff_dim=2048, dropout=0.1, norm=None, n_mels=80):
        super().__init__()
        self.linear_in = nn.Linear(n_mels, embed_dim)
        self.linear_out = nn.Linear(embed_dim, output_dim)

        self.positional_encoding = PositionalEncoding(embed_dim)
        
        self.decoders = torch.nn.ModuleList([
            TransformerDecoderLayer(
                embed_dim,
                n_heads,
                ff_dim,
                dropout,
                norm
            ) for _ in range(n_blocks)
        ])

    def forward(self, x, mem):
        x = self.linear_in(x)
        segment = x
        segment = self.positional_encoding(segment)
        for layer in self.decoders:
            x = layer(x, mem)
        x = self.linear_out(x)
        return x

class BaseEncoderLayer(nn.Module):
    def __init__(self, input_dim, embed_dim, n_heads=4, ff_dim=2048, dropout=0.1, norm=None):
        super().__init__()

    def doNorm(self, x):
        if self.norm is not None:
            x = self.norm(x)
        return x

    def plot_attention_maps(self, input_data, attn_maps, idx=0):
        if input_data is not None:
            input_data = input_data[idx].detach().cpu().numpy()
        else:
            input_data = np.arange(attn_maps[0][idx].shape[-1])
        attn_maps = [m[idx].detach().cpu().numpy() for m in attn_maps]

        num_heads = attn_maps[0].shape[0]
        num_layers = len(attn_maps)
        seq_len = input_data.shape[0]
        fig_size = 4 if num_heads == 1 else 3
        fig, ax = plt.subplots(num_layers, num_heads, figsize=(num_heads*fig_size, num_layers*fig_size))
        if num_layers == 1:
            ax = [ax]
        if num_heads == 1:
            ax = [[a] for a in ax]
        for row in range(num_layers):
            for column in range(num_heads):
                ax[row][column].imshow(attn_maps[row][column], origin='lower', vmin=0)
                ax[row][column].set_xticks(list(range(seq_len)))
                ax[row][column].set_xticklabels(input_data.tolist())
                ax[row][column].set_yticks(list(range(seq_len)))
                ax[row][column].set_yticklabels(input_data.tolist())
                ax[row][column].set_title(f"Layer {row+1}, Head {column+1}")
        fig.subplots_adjust(hspace=0.5)
        plt.show()
        return

class TransformerEncoderLayer(BaseEncoderLayer):
    def __init__(self, embed_dim, n_heads=4, ff_dim=2048, dropout=0.1, norm=None):
        super().__init__(embed_dim, n_heads, ff_dim, dropout, norm)
        self.norm = norm
        self.attn = MultiHeadedAttention(embed_dim, n_heads=n_heads, dropout=dropout)
        self.ff = PositionWiseFeedForward(embed_dim, ff_dim, dropout)

    def forward(self,x):
        #x[B, t, f]
        x = x + self.attn(x)
        x = self.doNorm(x)
        x = x + self.ff(x)
        x = self.doNorm(x)
        return x

class TransformerDecoderLayer(BaseEncoderLayer):
    def __init__(self, embed_dim, n_heads=4, ff_dim=2048, dropout=0.1, norm=None):
        super().__init__(embed_dim, n_heads, ff_dim, dropout, norm)
        self.n_heads = n_heads
        self.norm = norm
        self.attn1 = MultiHeadedAttention(embed_dim, n_heads=n_heads, dropout=dropout)
        self.attn2 = MultiHeadedAttention(embed_dim, n_heads=n_heads, dropout=dropout)
        self.ff = PositionWiseFeedForward(embed_dim, ff_dim, dropout)

    def make_attention_mask(self, input_dim, n_heads):
        head_dim = input_dim // n_heads
        mask = torch.tensor([1])
        mask = mask.repeat(head_dim, head_dim)
        mask = torch.tril(mask)
        return mask

    def forward(self,x, memory, useMask=False):
        #x[B, t, f]
        mask = None
        if useMask:
            mask = self.make_attention_mask(x.size(-2), self.n_heads)

        x = x + self.attn1(x, mask=mask)
        x = self.doNorm(x)
        x = x + self.attn2(x, memory=memory)
        x = self.doNorm(x)
        x = x + self.ff(x)
        x = self.doNorm(x)
        return x

class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_dim, ff_dim=2048, dropout=0.1):
        super().__init__()
        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(ff_dim, input_dim)
        )

    def forward(self, x):
        return self.linear_net(x)

class PositionalEncoding(nn.Module):
    #Calculated as sin(pos/10000^i/d) resp cos(pos/10000^((i-1)/d))) for odd numbers
    #we use absolute encoding here
    #x [B, T, f]
    def __init__(self, embed_dim, max_len=5000, dropout=0.0):
        super().__init__()
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(self.max_len, embed_dim, requires_grad=False)
        positions = torch.arange(0, self.max_len).unsqueeze(1).float()
        denominator = torch.exp(
            torch.arange(0, embed_dim, 2).float()
            * -(math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(positions * denominator)
        pe[:, 1::2] = torch.cos(positions * denominator)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self,x):
        x =  x + self.pe[ : , :x.size(-2)].clone().detach()
        return self.dropout(x)

    def plot_positional_encoding(self, full=None):
        #sns.set_theme()
        pe = self.pe.squeeze().T
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,3))
        pos = ax.imshow(pe, cmap="RdGy", extent=(1,pe.shape[1]+1,pe.shape[0]+1,1))
        fig.colorbar(pos, ax=ax)
        ax.set_xlabel("Position in sequence")
        ax.set_ylabel("Hidden dimension")
        ax.set_title("Positional encoding over hidden dimensions")
        ax.set_xticks([1]+[i*10 for i in range(1,1+pe.shape[1]//10)])
        ax.set_yticks([1]+[i*10 for i in range(1,1+pe.shape[0]//10)])
        plt.show()

        if full:
            fig, ax = plt.subplots(2, 2, figsize=(12,4))
            ax = [a for a_list in ax for a in a_list]
            for i in range(len(ax)):
                ax[i].plot(np.arange(1,17), pe[i,:16], color=f'C{i}', marker="o", markersize=6, markeredgecolor="black")
                ax[i].set_title(f"Encoding in hidden dimension {i+1}")
                ax[i].set_xlabel("Position in sequence", fontsize=10)
                ax[i].set_ylabel("Positional encoding", fontsize=10)
                ax[i].set_xticks(np.arange(1,17))
                ax[i].tick_params(axis='both', which='major', labelsize=10)
                ax[i].tick_params(axis='both', which='minor', labelsize=8)
                ax[i].set_ylim(-1.2, 1.2)
            fig.subplots_adjust(hspace=0.8)
            #sns.reset_orig()
            plt.show()

class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding module.
    See : Appendix B in https://arxiv.org/abs/1901.02860
    """
    def __init__(self, embed_dim, max_len = 50000, dropout_rate=0.0):
        super().__init__(embed_dim, dropout_rate, max_len)

    def forward(self, x, offset = 0):
        assert offset + x.size(-2) < self.max_len
        self.pe = self.pe.to(x.device)
        x = x * self.xscale
        pos_emb = self.pe[:, offset:offset + x.size(-2)]
        x =  x + pos_emb.clone().detach()
        return self.dropout(x)

class MultiHeadedAttention(nn.Module):
    def __init__(self, embed_dim, n_heads=4, dropout=0.1):
        super().__init__()
        assert embed_dim % n_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.dropout = nn.Dropout(p=dropout)

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(embed_dim, 3*embed_dim)

        #used for first mha in decoder
        self.kv_proj = nn.Linear(embed_dim, 2*embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)

        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def forward(self, x, mask=None, memory=None, return_attention=False):
        #x[B, f, t]
        batch_size, input_dim, embed_dim = x.size()

        if memory is not None:
            q = self.q_proj(x) # Separate query projection for decoder
            kv = self.kv_proj(memory)
            qkv = torch.cat((q,kv), -1)
            #k, v = self.kv_proj(memory).chunk(2, dim=-1)
        else:
            qkv = self.qkv_proj(x)
            #q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, input_dim, self.n_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, input_dim, embed_dim/nheads]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = self.scaled_dot_product(q, k, v, mask)
        values = values.permute(0, 2, 1, 3) # [Batch, input_dim, Head, embed_dim/nheads]
        values = values.reshape(batch_size, input_dim, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o
    
    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    #calculated as Attention(Q,K,V) = softmax(QK.T/d)V or V@a[K.T@Q] depends on which dimension is embed_dim and Q K are calculated out of it
    def scaled_dot_product(self, q, k, v, mask=None):
        embed_dim = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(embed_dim)
        if mask is not None:
            mask = mask.unsqueeze(0).unsqueeze(1).repeat(q.size(0),self.n_heads, 1, 1)
            attn_logits = attn_logits.masked_fill(mask == 0, -float('inf'))
        attn = F.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)
        values = torch.matmul(attn, v)
        return values, attn