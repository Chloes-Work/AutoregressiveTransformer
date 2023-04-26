import torch
import torch.nn as nn
from speechbrain.lobes.models.ECAPA_TDNN import BatchNorm1d

from Modules.Transformer.StatisticsPooling import AttentiveStatisticsPooling
from Modules.Transformer.Transformer import TransformerEncoder

class VanillaTransformerEncoder(nn.Module):
    def __init__(self, output_dim=512, embed_dim=512, n_blocks = 6, n_heads=4, ff_dim=2048, dropout=0.1, norm=None, n_mels=80):
        super().__init__()
        self.encoder = TransformerEncoder(output_dim,  embed_dim, n_blocks, n_heads, ff_dim, dropout, norm, n_mels=n_mels)
        
        self.pooling = AttentiveStatisticsPooling(output_dim)
        self.bn = BatchNorm1d(input_size=output_dim*2)
        self.fc = torch.nn.Linear(output_dim * 2, embed_dim)

    def forward(self,x):
        x = self.encoder(x)
        x = x.permute(0, 2, 1)
        x = self.pooling(x)
        x = self.bn(x)
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        x = x.squeeze()
        return x
