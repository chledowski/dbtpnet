import os

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode

from src.config import Config
from src.models.cnn_3d import CNN3D
from src.visualize import visualize

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)


@hydra.main(config_path="conf", config_name="config", version_base="1.2.0")
def main(cfg):
    print("Output path: ", os.getcwd())

    # Get DBT image from numpy
    if cfg.dataset.dbt_sample_path is not None:
        dbt_image = np.load(cfg.dataset.dbt_sample_path)
    else:
        raise ValueError("DBT sample path not provided")

    # Visualize 0th, 14th and 29th slice of the DBT image
    visualize(dbt_image, slice=0, save_path="dbt_image_0.png")
    visualize(dbt_image, slice=14, save_path="dbt_image_14.png")
    visualize(dbt_image, slice=29, save_path="dbt_image_29.png")

    # initialize the model
    model = CNN3D(cfg=cfg.model_3d,
                  final_shape=tuple(cfg.dataset.shape))

    # Load weights
    state_dict = torch.load(cfg.model_3d.pretrained_path, map_location=torch.device('cpu'))['state_dict']
    state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    # transform the image (please make sure your normalization values are correct)
    mean = cfg.dataset.normalization_mean
    std = cfg.dataset.normalization_std
    dbt_image = transform_image(dbt_image, mean, std, tuple(cfg.dataset.shape))

    # run the model
    synthetic_2d = model(dbt_image)

    # visualize the output
    visualize(synthetic_2d.detach().numpy(), save_path="synthetic_2d.png")


def transform_image(image, mean=212.8, std=184.7, shape=(1536, 1024)):
    """Transform image to fit the model input shape

    First, adds batch and channel dimensions.

    Then, resizes the image to the model input size (1536, 1024).

    Finally, normalizes the image, using the mean and std values from the
    NYU dataset (mean: 212.8, std: 184.7)
    """
    # convert to torch
    image = torch.from_numpy(image).float()

    # add batch and channel dimensions
    image = image.unsqueeze(0).unsqueeze(0)

    # resize to model input size, keep the depth dimension ([1, 1, d, w, h] -> [1, 1, d, 1536, 1024])
    resizer = Resize(shape, interpolation=InterpolationMode.BILINEAR)

    # Initialize an empty list to store the resized slices
    resized_slices = []
    depth = image.shape[2]

    # Loop through each depth slice and apply resizing
    for i in range(depth):
        current_slice = image[0, 0, i]
        current_slice = current_slice.unsqueeze(0)  # add a channel dimension
        resized_slice = resizer(current_slice)
        resized_slices.append(resized_slice.unsqueeze(0))  # add a depth dimension

    # Stack the slices along the depth dimension
    resized_image = torch.stack(resized_slices, dim=2)

    # normalize
    normalized_image = (resized_image - mean) / std

    return normalized_image


if __name__ == "__main__":
    main()
