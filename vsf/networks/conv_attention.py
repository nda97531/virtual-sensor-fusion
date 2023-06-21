from typing import Union
import torch as tr
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv1dBlock


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio: Union[float, int] = 16, pool_types=('avg', 'max')):
        super(ChannelGate, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(gate_channels, int(gate_channels // reduction_ratio)),
            nn.ReLU(),
            nn.Linear(int(gate_channels // reduction_ratio), gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.adaptive_max_pool1d(x, 1).squeeze(-1)
                channel_att_raw = self.mlp(max_pool)
            else:
                raise ValueError("pool_types must be avg/max")

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = tr.sigmoid(channel_att_sum).unsqueeze(-1)
        return x * scale


class SpatialGate(nn.Module):
    def __init__(self, input_len, kernel_size=7, conv_norm: str = None):
        super(SpatialGate, self).__init__()

        self.spatial = Conv1dBlock(
            input_len=input_len,
            in_filters=2, out_filters=1,
            kernel_size=kernel_size, padding=(kernel_size - 1) // 2, activation=nn.Identity(),
            norm_layer=conv_norm
        )

    def forward(self, x):
        # input shape: [batch, channel, time]

        x_compress = tr.cat([tr.max(x, 1, keepdim=True)[0], tr.mean(x, 1, keepdim=True)], dim=1)  # channel pooling
        # shape: [batch, 2, time]

        x_out = self.spatial(x_compress)
        scale = tr.sigmoid(x_out)  # broadcasting
        # shape: [batch, 1, time]

        return x * scale  # shape: [batch, channel, time]


class CBAM(nn.Module):
    def __init__(self, channel_gate: ChannelGate = None, spatial_gate: SpatialGate = None):
        assert (channel_gate is not None) or (spatial_gate is not None)
        super().__init__()

        att = []
        if channel_gate is not None:
            att.append(channel_gate)
        if spatial_gate is not None:
            att.append(spatial_gate)

        self.att = nn.Sequential(*att)

    def forward(self, x):
        x_out = self.att(x)
        return x_out
