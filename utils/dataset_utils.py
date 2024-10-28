'''
Helper functions for loading datasets
'''

import h5py
import zarr
import torch
import os
from PIL import Image
import numpy as np
import concurrent.futures


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
    
def hdf5_to_zarr_parallel(hdf5_path, max_workers=64):
    '''
    Function for duplicating an existing hdf5 file without a duplicate zarr file and saving it as a zarr file.
    Parallelized with configurable number of workers.
    '''
    # Determine zarr path
    zarr_path = hdf5_path.replace('.hdf5', '.zarr')

    # Open HDF5 file
    with h5py.File(hdf5_path, 'r') as hdf5_file:
        # Create Zarr store
        store = zarr.DirectoryStore(zarr_path)
        root = zarr.group(store=store, overwrite=True)

        def copy_dataset(item, zarr_node, key):
            ''' Function to copy a single dataset in parallel '''
            compression = item.compression if item.compression else 'blosc'
            compression_opts = item.compression_opts if item.compression_opts else None
            chunks = item.chunks if item.chunks else (1000,)
            zarr_node.create_dataset(key, data=item[:], chunks=chunks, dtype=item.dtype, compression=compression, compression_opts=compression_opts)

        def copy_node(hdf5_node, zarr_node):
            ''' Recursively copy HDF5 groups and datasets to Zarr '''
            futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                for key, item in hdf5_node.items():
                    if isinstance(item, h5py.Group):
                        zarr_group = zarr_node.require_group(key)
                        # Recursively copy groups
                        futures.append(executor.submit(copy_node, item, zarr_group))
                    elif isinstance(item, h5py.Dataset):
                        # Copy datasets in parallel
                        futures.append(executor.submit(copy_dataset, item, zarr_node, key))
                # Wait for all futures to complete
                concurrent.futures.wait(futures)

        # Copy entire HDF5 file structure to Zarr file in parallel
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
    
    
import matplotlib.pyplot as plt

def visualize_voxel(voxels):
    if isinstance(voxels, torch.Tensor):
        voxels = voxels.cpu().numpy()

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    indices = np.argwhere(voxels[0] != 0)
    colors = voxels[1:, indices[:, 0], indices[:, 1], indices[:, 2]].T

    ax.scatter(indices[:, 0], indices[:, 1], indices[:, 2], c=colors/255., marker='s')

    # Set labels and show the plot
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 64)
    ax.set_zlim(0, 64)  

    plt.show()
    
    
import cv2

def save_video_from_array(frames):
    fps = 20  # Frames per second
    height, width = frames.shape[1], frames.shape[2]
    video_filename = '/home/yilong/Desktop/output_video.mp4'

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

    # Iterate over the frames and write them to the video
    for i in range(frames.shape[0]):
        frame = frames[i]
        out.write(frame)

    # Release the video writer object
    out.release()

    print(f'Video saved as {video_filename}')
    

def segment_color_object(image, color='green', threshold=150):
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Define color ranges in HSV
    if color == 'green':
        lower_color = np.array([40, 40, threshold])
        upper_color = np.array([80, 255, 255])
    elif color == 'red':
        # Red has two ranges in HSV because it wraps around the hue circle
        lower_red1 = np.array([0, 40, threshold])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 40, threshold])
        upper_red2 = np.array([180, 255, 255])
        
        # Create two masks for red
        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
    elif color == 'blue':
        lower_color = np.array([100, 40, threshold])
        upper_color = np.array([140, 255, 255])
        mask = cv2.inRange(hsv_image, lower_color, upper_color)
    elif color == 'cyan':
        lower_color = np.array([80, 40, threshold])
        upper_color = np.array([100, 255, 255])
        mask = cv2.inRange(hsv_image, lower_color, upper_color)
    else:
        raise ValueError("Color must be one of 'red', 'green', 'blue', or 'cyan'")
    
    if color != 'red':
        # For green, blue, and cyan, we only need one mask
        mask = cv2.inRange(hsv_image, lower_color, upper_color)
    
    # Apply the mask to the original image
    segmented_image = cv2.bitwise_and(image, image, mask=mask)
    
    return segmented_image, mask

def save_combined_image(original_image, segmented_image, mask, directory='debug', combined_name='combined_image'):
    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Create a figure to place the images side by side
    plt.figure(figsize=(15, 5))
    
    # Display the original image
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Display the segmented image
    plt.subplot(1, 3, 2)
    plt.imshow(segmented_image)
    plt.title('Segmented Image')
    plt.axis('off')
    
    # Display the mask
    plt.subplot(1, 3, 3)
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')
    plt.axis('off')
    
    # Save the combined image
    combined_image_path = os.path.join(directory, f'{combined_name}.png')
    plt.savefig(combined_image_path, bbox_inches='tight')
    plt.close()
    
    print(f'Combined image saved to: {combined_image_path}')