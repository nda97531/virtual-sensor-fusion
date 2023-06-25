import math

import torch as tr
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from .conv import Conv1dBlock
from .conv_attention import CBAM, SpatialGate, ChannelGate


def flatten_by_gap(x):
    """
    Flatten a tensor by global pooling average over the last channel
    Args:
        x: tensor shape [batch size, channel, window length]

    Returns:
        tensor shape [batch size, channel]
    """
    return F.adaptive_avg_pool1d(x, 1).squeeze(-1)


def take_last_time_step(x):
    """
    Flatten a tensor by taking only the last time step
    Args:
        x: tensor shape [batch size, channel, window length]

    Returns:
        tensor shape [batch size, channel]
    """
    return x[:, :, -1]


class TCN(nn.Module):
    def __init__(self,
                 input_shape: tuple,
                 how_flatten: str,
                 n_tcn_channels: tuple = (64,) * 5 + (128,) * 2,
                 tcn_kernel_size: int = 2,
                 dilation_base: int = 2,
                 tcn_drop_rate: float = 0.2,
                 use_spatial_dropout: bool = False,
                 conv_norm: str = 'batch',
                 attention_conv_norm: str = 'batch'
                 ):
        """
        TCN backbone

        Args:
            input_shape: a tuple showing input shape to the model [window size, channel]
            how_flatten: how to flatten feature vectors after TCN; choices are 'last time step'/'gap'/'attention gap'/
                'channel attention gap'/'spatial attention gap'
            n_tcn_channels: tuple of dim for each res-block in TCN
            tcn_kernel_size: TCN kernel size
            dilation_base: TCN dilation base; block number i will have dilation of `dilation_base`**i
            tcn_drop_rate: dropout rate in TCN
            use_spatial_dropout: whether to use spatial dropout (entire channel) or normal dropout
            conv_norm: type of norm after conv layers
            attention_conv_norm: type of norm after conv layers in the attention block
        """
        super().__init__()

        self.B = len(n_tcn_channels)
        self.N = 2
        self.k = tcn_kernel_size

        rec_field = self.receptive_field()
        if rec_field < input_shape[0]:
            logger.warning(f'Receptive field is smaller than input size: {rec_field} < {input_shape[0]}')
        else:
            logger.info(f'Window size: {input_shape[0]}; Receptive field: {rec_field}')

        layers = []
        for i in range(len(n_tcn_channels)):
            dilation_rate = dilation_base ** i
            in_channels = input_shape[1] if i == 0 else n_tcn_channels[i - 1]
            out_channels = n_tcn_channels[i]

            # only drop between res blocks, don't drop after the last block
            block_dropout_rate = 0. if (i == len(n_tcn_channels) - 1) else tcn_drop_rate

            layers.append(ResTempBlock(
                input_len=input_shape[0],
                n_channels_in=in_channels,
                n_channels_out=out_channels,
                kernel_size=tcn_kernel_size,
                dilation=dilation_rate,
                dropout_rate=block_dropout_rate,
                use_spatial_dropout=use_spatial_dropout,
                n_conv_layers=2,
                conv_norm=conv_norm,
            ))

        self.feature_extractor = nn.Sequential(*layers)

        if how_flatten == "last time step":
            self.cbam = nn.Identity()
            self.flatten = take_last_time_step
        elif how_flatten == "gap":
            self.cbam = nn.Identity()
            self.flatten = flatten_by_gap
        elif how_flatten == "channel attention gap":
            self.cbam = CBAM(channel_gate=ChannelGate(
                gate_channels=n_tcn_channels[-1],
                reduction_ratio=math.sqrt(n_tcn_channels[-1])
            ))
            self.flatten = flatten_by_gap
        elif how_flatten == "spatial attention gap":
            self.cbam = CBAM(spatial_gate=SpatialGate(
                input_len=input_shape[0],
                conv_norm=attention_conv_norm
            ))
            self.flatten = flatten_by_gap
        elif how_flatten == "attention gap":
            self.cbam = CBAM(
                channel_gate=ChannelGate(
                    gate_channels=n_tcn_channels[-1],
                    reduction_ratio=math.sqrt(n_tcn_channels[-1])
                ),
                spatial_gate=SpatialGate(
                    input_len=input_shape[0],
                    conv_norm=attention_conv_norm
                )
            )
            self.flatten = flatten_by_gap
        else:
            raise ValueError("how_flatten must be 'last time step'/'gap'/'attention gap'/"
                             "'channel attention gap'/'spatial attention gap'")

    def forward(self, x):
        """
        Forward function

        Args:
            x: tensor shape [batch, channel, time step]

        Returns:
            tensor shape [batch, channel]
        """
        x = self.feature_extractor(x)
        # batch size, feature, time step

        x = self.cbam(x)
        x = self.flatten(x)
        # batch size, feature

        return x

    def receptive_field(self) -> int:
        """
        Calculate receptive field of TCN

        Returns:
            receptive field (scalar)
        """
        # k: kernel size
        # B: number of blocks
        # N: number of TCN layers per block
        return 1 + self.N * (self.k - 1) * (2 ** self.B - 1)


class ResTempBlock(nn.Module):
    def __init__(self, input_len, n_channels_in, n_channels_out, kernel_size, dilation, dropout_rate=0.2,
                 use_spatial_dropout=False, n_conv_layers=2, conv_norm: str = 'batch'):
        """
        A residual block in TCN

        Args:
            input_len: input length (time step)
            n_channels_in: number of input features
            n_channels_out: number of output features
            kernel_size: conv kernel size
            dilation: conv dilation
            dropout_rate: drop out rate
            use_spatial_dropout: whether to use spatial dropout or normal dropout
            n_conv_layers: number of conv layers in this res-block
            conv_norm: type of norm after each conv layer, can be 'batch', 'layer' or empty
        """
        super().__init__()

        org_channel_in = n_channels_in

        block = []
        for i in range(n_conv_layers):
            block.append(Conv1dBlock(
                input_len=input_len,
                in_filters=n_channels_in,
                out_filters=n_channels_out,
                kernel_size=kernel_size,
                stride=1,
                padding=((kernel_size - 1) * dilation, 0),
                dilation=dilation,
                drop_rate=dropout_rate,
                use_spatial_dropout=use_spatial_dropout,
                norm_layer=conv_norm
            ))
            if i == 0:
                n_channels_in = n_channels_out

        self.block = nn.Sequential(*block)
        self.downsample = nn.Conv1d(org_channel_in, n_channels_out, 1) if org_channel_in != n_channels_out else None

    def forward(self, x):
        out = self.block(x)
        res = x if self.downsample is None else self.downsample(x)
        return tr.relu(out + res)
