import PIL.Image
import numpy as np


def visualize_3d(image, slice=0, save_path=None):
    image_slice = image[slice]
    image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min())
    image_slice = (image_slice * 255).astype(np.uint8)
    pil_image = PIL.Image.fromarray(image_slice)
    pil_image.save(save_path)


def visualize_2d(image, save_path=None):
    image = np.clip(image, -1.15, 3.99)  # clip to remove outliers coming from NN noise
    image = (image - image.min()) / (image.max() - image.min())
    image = (image * 255).astype(np.uint8)
    pil_image = PIL.Image.fromarray(image)
    pil_image.save(save_path)


def visualize(image, slice=0, save_path=None):
    # reduce batch/channel dimensions if needed
    while len(image.shape) > 3:
        image = image[0]

    # visualize 3D image
    if len(image.shape) == 3 and image.shape[0] > 1:
        visualize_3d(image, slice, save_path)

    elif len(image.shape) == 3 and image.shape[0] == 1:
        visualize_2d(image[0], save_path)

    # visualize 2D image
    elif len(image.shape) == 2:
        visualize_2d(image, save_path)

    else:
        raise ValueError("Invalid image shape: {}".format(image.shape))
