'''
Helper functions for loading datasets
'''

import h5py
import zarr
import torch
import os
from PIL import Image
import numpy as np


def hdf5_to_zarr(hdf5_path):
    '''
    Function for duplicating an existing hdf5 file without a duplicate zarr file and saving it as a zarr file.
    '''
    # determine zarr path
    zarr_path = hdf5_path.replace('.hdf5', '.zarr')

    # open hdf5 file
    with h5py.File(hdf5_path, 'r') as hdf5_file:
        # create zarr store
        store = zarr.DirectoryStore(zarr_path)
        root = zarr.group(store=store, overwrite=True)
        
        def copy_node(hdf5_node, zarr_node):
            if isinstance(hdf5_node, h5py.Group):
                for key, item in hdf5_node.items():
                    if isinstance(item, h5py.Group):
                        zarr_group = zarr_node.require_group(key)
                        copy_node(item, zarr_group)
                    elif isinstance(item, h5py.Dataset):
                        compression = item.compression if item.compression else 'blosc'
                        compression_opts = item.compression_opts if item.compression_opts else None
                        chunks = item.chunks if item.chunks else (1000,)
                        zarr_node.create_dataset(key, data=item[:], chunks=chunks, dtype=item.dtype, compression=compression, compression_opts=compression_opts)
            elif isinstance(hdf5_node, h5py.Dataset):
                compression = hdf5_node.compression if hdf5_node.compression else 'blosc'
                compression_opts = hdf5_node.compression_opts if hdf5_node.compression_opts else None
                chunks = hdf5_node.chunks if hdf5_node.chunks else (1000,)
                zarr_node.create_dataset(name=hdf5_node.name, data=hdf5_node[:], chunks=chunks, dtype=hdf5_node.dtype, compression=compression, compression_opts=compression_opts)

        # copy entire hdf5 file structure to zarr file
        copy_node(hdf5_file, root)

    print(f'Duplicated {hdf5_path} as zarr file {zarr_path}')

def save_consecutive_images(tensor, save_path="debug/combined_image.png"):
    # Ensure the save path directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Ensure the tensor shape is [6, 128, 128]
    if tensor.shape != torch.Size([6, 128, 128]):
        raise ValueError("Expected tensor shape is [6, 128, 128], but got {}".format(tensor.shape))

    # Split the tensor into two images of shape [3, 128, 128] each
    image1 = tensor[:3]  # First 3 channels
    image2 = tensor[3:]  # Last 3 channels

    # Convert the tensor values to integers in the range [0, 255]
    image1 = image1.mul(255).byte().numpy()
    image2 = image2.mul(255).byte().numpy()

    # Convert to shape (128, 128, 3) for PIL (H, W, C)
    image1 = np.transpose(image1, (1, 2, 0))
    image2 = np.transpose(image2, (1, 2, 0))

    # Convert numpy arrays to PIL images
    pil_image1 = Image.fromarray(image1)
    pil_image2 = Image.fromarray(image2)

    # Combine the two images side by side
    combined_image = Image.new('RGB', (256, 128))  # Width: 128 + 128, Height: 128
    combined_image.paste(pil_image1, (0, 0))
    combined_image.paste(pil_image2, (128, 0))

    # Save the combined image
    combined_image.save(save_path)
    print(f"Saved combined image to {save_path}")