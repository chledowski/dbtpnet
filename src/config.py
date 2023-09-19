from dataclasses import dataclass


@dataclass
class DatasetCfg:
    dbt_sample_path: str
    normalization_mean: float
    normalization_std: float
    shape: list


@dataclass
class Model3DCfg:
    pretrained_path: str
    first_3d_layer_kernel_size: list
    first_3d_layer_conv_stride: list
    rest_3d_layer_kernel_size: list
    rest_3d_layer_conv_stride: list
    apply_stride_every_second_cnn: bool
    num_3d_channels_list: list
    norm_class: str
    num_groups: int
    output_transform: str
    residual_connections: bool


# ======================================== #
# Config
@dataclass
class Config:
    dataset: DatasetCfg
    model_3d: Model3DCfg
