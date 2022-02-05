from typing import Any, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from monai.networks.blocks import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.convutils import same_padding
from utils.utils import calculate_out_shape, log_average


class Lambda(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class Encoder(nn.Module):
    """
    Big inspiration taken from MONAI's AutoEncoder's class
    """

    def __init__(
        self,
        spatial_dims: int,
        in_shape: Sequence[int],
        out_channels: int,
        latent_size: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        dilation: Union[Sequence[int], int] = 1,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        inter_channels: Optional[list] = None,
        inter_dilations: Optional[list] = None,
        num_inter_units: int = 2,
        act: Optional[Union[Tuple, str]] = Act.LEAKYRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: Optional[Union[Tuple, str, float]] = None,
        bias: bool = True,
        dimensions: Optional[int] = None,
    ) -> None:
        """
        Initialize the AutoEncoder.

        .. deprecated:: 0.6.0
            ``dimensions`` is deprecated, use ``spatial_dims`` instead.

        """
        super().__init__()
        self.dimensions = spatial_dims if dimensions is None else dimensions
        self.in_channels, *self.in_shape = in_shape
        self.out_channels = out_channels
        self.latent_size = latent_size
        self.channels = list(channels)
        self.strides = list(strides)
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.num_inter_units = num_inter_units
        self.inter_channels = inter_channels if inter_channels is not None else []
        self.inter_dilations = list(inter_dilations or [1] * len(self.inter_channels))

        # The number of channels and strides should match
        if len(channels) != len(strides):
            raise ValueError("Autoencoder expects matching number of channels and strides")

        # get final size
        self.encoded_channels = self.in_channels
        self.final_size = np.asarray(self.in_shape, dtype=int)
        padding = same_padding(self.kernel_size, dilation)
        for s in strides:
            self.final_size = calculate_out_shape(self.final_size, self.kernel_size, dilation, s, padding)  # type: ignore

        # layers
        self.encode, self.encoded_channels = self._get_encode_module(self.encoded_channels, channels, strides)
        linear_size = int(np.product(self.final_size)) * self.encoded_channels
        self.encodeL = nn.Linear(linear_size, self.latent_size)


    def _get_encode_module(
        self, in_channels: int, channels: Sequence[int], strides: Sequence[int]
    ) -> Tuple[nn.Sequential, int]:
        encode = nn.Sequential()
        layer_channels = in_channels

        for i, (c, s) in enumerate(zip(channels, strides)):
            layer = self._get_encode_layer(layer_channels, c, s, False)
            encode.add_module("encode_%i" % i, layer)
            layer_channels = c

        return encode, layer_channels


    def _get_encode_layer(self, in_channels: int, out_channels: int, strides: int, is_last: bool) -> nn.Module:

        mod: nn.Module
        if self.num_res_units > 0:
            mod = ResidualUnit(
                spatial_dims=self.dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_last,
            )
        mod = Convolution(
            dimensions=self.dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_last,
        )
        return mod


    def forward(self, x: torch.Tensor) -> Any:
        x = self.encode(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.encodeL(x))
        return x


class Decoder(nn.Module):
    """
    Big inspiration taken from MONAI's AutoEncoder's class
    """

    def __init__(
        self,
        spatial_dims: int,
        in_shape: Sequence[int],
        out_channels: int,
        final_size: Sequence[int],
        latent_size: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        dilation: Union[Sequence[int], int] = 1,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        inter_channels: Optional[list] = None,
        inter_dilations: Optional[list] = None,
        num_inter_units: int = 2,
        act: Optional[Union[Tuple, str]] = Act.LEAKYRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: Optional[Union[Tuple, str, float]] = None,
        bias: bool = True,
        dimensions: Optional[int] = None,
    ) -> None:
        """
        Initialize the AutoEncoder.

        .. deprecated:: 0.6.0
            ``dimensions`` is deprecated, use ``spatial_dims`` instead.

        """
        super().__init__()
        self.dimensions = spatial_dims if dimensions is None else dimensions
        self.in_channels, *self.in_shape = in_shape
        self.out_channels = out_channels
        self.latent_size = latent_size
        self.channels = list(channels)
        self.strides = list(strides)
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.num_inter_units = num_inter_units
        self.inter_channels = inter_channels if inter_channels is not None else []
        self.inter_dilations = list(inter_dilations or [1] * len(self.inter_channels))

        # The number of channels and strides should match
        if len(channels) != len(strides):
            raise ValueError("Autoencoder expects matching number of channels and strides")

        # get final size
        self.encoded_channels = channels[-1]
        self.final_size = final_size
        # padding = same_padding(self.kernel_size, dilation)
        # for s in strides:
        #     self.final_size = calculate_out_shape(self.final_size, self.kernel_size, dilation, s, padding)  # type: ignore

        linear_size = int(np.product(self.final_size)) * self.encoded_channels
        decode_channel_list = list(channels[-2::-1]) + [out_channels]

        # layers
        self.decode, _ = self._get_decode_module(self.encoded_channels, decode_channel_list, strides[::-1] or [1])
        self.decodeL = nn.Linear(self.latent_size, linear_size)


    def _get_decode_module(
        self, in_channels: int, channels: Sequence[int], strides: Sequence[int]
    ) -> Tuple[nn.Sequential, int]:
        """
        Returns the decode part of the network by building up a sequence of layers returned by `_get_decode_layer`.
        """
        decode = nn.Sequential()
        layer_channels = in_channels

        for i, (c, s) in enumerate(zip(channels, strides)):
            layer = self._get_decode_layer(layer_channels, c, s, i == (len(strides) - 1))
            decode.add_module("decode_%i" % i, layer)
            layer_channels = c

        return decode, layer_channels


    def _get_decode_layer(self, in_channels: int, out_channels: int, strides: int, is_last: bool) -> nn.Sequential:
        """
        Returns a single layer of the decoder part of the network.
        """
        decode = nn.Sequential()

        # went from transposed conv to upsample following: https://distill.pub/2016/deconv-checkerboard/
        upsample = nn.Upsample(scale_factor=2, mode='nearest')

        decode.add_module("upsample", upsample)
        # 

        conv = Convolution(
            spatial_dims=self.dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            dilation=self.dilation,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_last and self.num_res_units == 0,
            is_transposed=False,
        )

        decode.add_module("conv", conv)

        if self.num_res_units > 0:
            ru = ResidualUnit(
                spatial_dims=self.dimensions,
                in_channels=out_channels,
                out_channels=out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_last,
            )

            decode.add_module("resunit", ru)

        return decode


    def forward(self, x: torch.Tensor, use_sigmoid: bool = True) -> Any:
        x = F.relu(self.decodeL(x))
        x = x.view(x.shape[0], self.encoded_channels, *self.final_size)
        x = self.decode(x)
        if use_sigmoid:
            x = torch.sigmoid(x)
        return x


class Discriminator(torch.nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_shape: Sequence[int],
        out_channels: int,
        latent_size: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        dilation: Union[Sequence[int], int] = 1
    ) -> None:
        super().__init__()

        self.main = torch.nn.Sequential(
            Encoder(spatial_dims=spatial_dims, 
                  in_shape=in_shape,
                  out_channels=out_channels,
                  latent_size=latent_size, 
                  channels=channels,
                  strides=strides,
                  kernel_size=kernel_size,
                  dilation=dilation),
            Lambda(lambda x: torch.mean(x, dim=1))
        )
        

    def forward(self, x):
        return self.main(x)


class EncoderVAE(nn.Module):
    """
    Big inspiration taken from MONAI's AutoEncoder's class
    """

    def __init__(
        self,
        spatial_dims: int,
        in_shape: Sequence[int],
        out_channels: int,
        latent_size: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        dilation: Union[Sequence[int], int] = 1,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        inter_channels: Optional[list] = None,
        inter_dilations: Optional[list] = None,
        num_inter_units: int = 2,
        act: Optional[Union[Tuple, str]] = Act.LEAKYRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: Optional[Union[Tuple, str, float]] = None,
        bias: bool = True,
        dimensions: Optional[int] = None,
    ) -> None:
        """
        Initialize the AutoEncoder.

        .. deprecated:: 0.6.0
            ``dimensions`` is deprecated, use ``spatial_dims`` instead.

        """
        super().__init__()
        self.dimensions = spatial_dims if dimensions is None else dimensions
        self.in_channels, *self.in_shape = in_shape
        self.out_channels = out_channels
        self.latent_size = latent_size
        self.channels = list(channels)
        self.strides = list(strides)
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.num_inter_units = num_inter_units
        self.inter_channels = inter_channels if inter_channels is not None else []
        self.inter_dilations = list(inter_dilations or [1] * len(self.inter_channels))

        # The number of channels and strides should match
        if len(channels) != len(strides):
            raise ValueError("Autoencoder expects matching number of channels and strides")

        # get final size
        self.encoded_channels = self.in_channels
        self.final_size = np.asarray(self.in_shape, dtype=int)
        padding = same_padding(self.kernel_size, dilation)
        for s in strides:
            self.final_size = calculate_out_shape(self.final_size, self.kernel_size, dilation, s, padding)  # type: ignore

        # layers
        self.encode, self.encoded_channels = self._get_encode_module(self.encoded_channels, channels, strides)
        linear_size = int(np.product(self.final_size)) * self.encoded_channels
        self.mu = nn.Linear(linear_size, self.latent_size)
        self.logvar = nn.Linear(linear_size, self.latent_size)


    def _get_encode_module(
        self, in_channels: int, channels: Sequence[int], strides: Sequence[int]
    ) -> Tuple[nn.Sequential, int]:
        encode = nn.Sequential()
        layer_channels = in_channels

        for i, (c, s) in enumerate(zip(channels, strides)):
            layer = self._get_encode_layer(layer_channels, c, s, False)
            encode.add_module("encode_%i" % i, layer)
            layer_channels = c

        return encode, layer_channels


    def _get_encode_layer(self, in_channels: int, out_channels: int, strides: int, is_last: bool) -> nn.Module:

        mod: nn.Module
        if self.num_res_units > 0:
            mod = ResidualUnit(
                spatial_dims=self.dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_last,
            )
        mod = Convolution(
            dimensions=self.dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_last,
        )
        return mod


    def forward(self, x: torch.Tensor) -> Any:
        x = self.encode(x)
        x = x.view(x.shape[0], -1)
        mu = F.relu(self.mu(x))
        logvar = F.relu(self.logvar(x))
        return mu, logvar
