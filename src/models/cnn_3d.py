import torch
from torch import nn
from config import Model3DCfg


class CNN3D(torch.nn.Module):
    def __init__(self, cfg: Model3DCfg, final_shape: tuple[int, int] = (2116, 1339)):
        super(CNN3D, self).__init__()

        self.cfg = cfg

        self.first_3d_layer_kernel_size = tuple(cfg.first_3d_layer_kernel_size)
        self.first_3d_layer_conv_stride = tuple(cfg.first_3d_layer_conv_stride)
        self.rest_3d_layer_kernel_size = tuple(cfg.rest_3d_layer_kernel_size)
        self.rest_3d_layer_conv_stride = tuple(cfg.rest_3d_layer_conv_stride)
        self.apply_stride_every_second_cnn = cfg.apply_stride_every_second_cnn
        self.num_3d_channels_list = cfg.num_3d_channels_list
        self.norm_class = cfg.norm_class
        self.num_groups = cfg.num_groups
        self.output_transform = cfg.output_transform
        self.residual_connections = cfg.residual_connections

        self.first_conv3d = nn.Conv3d(
            1,
            self.num_3d_channels_list[0],
            kernel_size=self.first_3d_layer_kernel_size,
            stride=self.first_3d_layer_conv_stride,
            padding=tuple(x // 2 for x in self.first_3d_layer_kernel_size),
        )

        self.first_downsample = nn.Conv3d(1, self.num_3d_channels_list[0], kernel_size=1, stride=self.first_3d_layer_conv_stride, padding=(0, 0, 0))

        self.rest_conv3d_layers = nn.ModuleList()
        self.rest_bn3d_layers = nn.ModuleList()
        self.rest_downsample_layers = nn.ModuleList()

        for i in range(len(self.num_3d_channels_list) - 1):
            if self.apply_stride_every_second_cnn and i % 2 == 0:
                self.rest_conv3d_layers.append(
                    nn.Conv3d(
                        self.num_3d_channels_list[i],
                        self.num_3d_channels_list[i + 1],
                        kernel_size=self.rest_3d_layer_kernel_size,
                        stride=(1, 1, 1),
                        padding=tuple(x // 2 for x in self.rest_3d_layer_kernel_size),
                    )
                )
                self.rest_downsample_layers.append(
                    nn.Conv3d(self.num_3d_channels_list[i], self.num_3d_channels_list[i + 1], kernel_size=1, stride=(1, 1, 1), padding=(0, 0, 0))
                )
            else:
                self.rest_conv3d_layers.append(
                    nn.Conv3d(
                        self.num_3d_channels_list[i],
                        self.num_3d_channels_list[i + 1],
                        kernel_size=self.rest_3d_layer_kernel_size,
                        stride=self.rest_3d_layer_conv_stride,
                        padding=tuple(x // 2 for x in self.rest_3d_layer_kernel_size),
                    )
                )
                self.rest_downsample_layers.append(
                    nn.Conv3d(
                        self.num_3d_channels_list[i], self.num_3d_channels_list[i + 1], kernel_size=1, stride=self.rest_3d_layer_conv_stride, padding=(0, 0, 0)
                    )
                )
            self.rest_bn3d_layers.append(resolve_norm_layer(self.num_3d_channels_list[i + 1], self.norm_class, self.num_groups))

        self.compress_conv = nn.Conv3d(self.num_3d_channels_list[-1], 1, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))

        if self.cfg.pooling == "max":
            self.pool = nn.AdaptiveMaxPool3d((1, final_shape[0], final_shape[1]))
        elif self.cfg.pooling == "avg":
            self.pool = nn.AdaptiveAvgPool3d((1, final_shape[0], final_shape[1]))

        if self.output_transform == "layer":
            self.layer_norm = nn.LayerNorm([final_shape[0], final_shape[1]])

        self.relu = nn.ReLU()

        self.initialize()
        self.parameters()

    def freeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
        print(f"Frozen CNN3D converter weights.")

    def initialize(self):
        nn.init.kaiming_normal_(self.first_conv3d.weight, nonlinearity="relu")
        for i, (conv_layer, bn_layer) in enumerate(zip(self.rest_conv3d_layers, self.rest_bn3d_layers)):
            nn.init.kaiming_normal_(conv_layer.weight, nonlinearity="relu")
            nn.init.constant_(bn_layer.weight, 1)
            nn.init.constant_(bn_layer.bias, 0)
        nn.init.kaiming_normal_(self.compress_conv.weight, nonlinearity="relu")

    @staticmethod
    def standardize(image):
        mean = torch.mean(image)
        std = torch.std(image)
        return ((image - mean) / (std + 1e-5)).to(dtype=torch.float32)

    def forward(self, x):
        residual = x
        x = self.first_conv3d(x)
        if self.residual_connections:
            x += self.first_downsample(residual)

        for i, (conv_layer, bn_layer, ds_layer) in enumerate(zip(self.rest_conv3d_layers, self.rest_bn3d_layers, self.rest_downsample_layers)):
            residual = x
            x = self.relu(bn_layer(conv_layer(x)))
            if self.residual_connections:
                x += ds_layer(residual)

        x = self.compress_conv(x)
        x = self.pool(x)
        x = x.squeeze(1).squeeze(1)  # squeezes channels and depth

        if self.output_transform == "layer":
            x = self.layer_norm(x)
        elif self.output_transform == "standardize":
            x = self.standardize(x)
        else:
            pass

        x = x.unsqueeze(1)
        return x


def resolve_norm_layer(planes, norm_class, num_groups=1):
    if norm_class.lower() == "batch":
        return nn.BatchNorm2d(planes)
    if norm_class.lower() == "group":
        return nn.GroupNorm(num_groups, planes)
    raise NotImplementedError(f"norm_class must be batch or group, but {norm_class} was given")
