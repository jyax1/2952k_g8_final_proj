'''
Helper functions for loading datasets
'''

import h5py
import zarr
from zarr import ZipStore
import torch
import os
from PIL import Image
import numpy as np
import concurrent.futures
from tqdm import tqdm
import importlib.resources


def hdf5_to_zarr(hdf5_path):
    '''
    Function for duplicating an existing hdf5 file without a duplicate zarr file and saving it as a zarr file.
    '''
    print(f'Converting {hdf5_path} to zarr file')
    
    hdf5_file = h5py.File(hdf5_path, 'r')
    zarr_path = hdf5_path.replace('.hdf5', '.zarr')
    root = zarr.open(zarr_path, mode='w')

    def copy_dataset(item, zarr_node, key):
        ''' Copy HDF5 dataset to Zarr '''
        compression = item.compression
        compression_opts = item.compression_opts
        chunks = item.chunks if item.chunks else (1000,)
        zarr_node.create_dataset(key, data=item[:], chunks=chunks, dtype=item.dtype, compression=compression, compression_opts=compression_opts)

    def copy_node(hdf5_node, zarr_node):
        ''' Recursively copy HDF5 groups and datasets to Zarr '''
        for key, item in hdf5_node.items():
            if isinstance(item, h5py.Group):
                zarr_group = zarr_node.require_group(key)
                # Recursively copy groups
                copy_node(item, zarr_group)
            elif isinstance(item, h5py.Dataset):
                # Copy datasets
                copy_dataset(item, zarr_node, key)

    # Copy entire HDF5 file structure to Zarr file
    copy_node(hdf5_file, root)

    print(f'Duplicated {hdf5_path} as zarr file {zarr_path}')
    
def hdf5_to_zarr_parallel(hdf5_path, max_workers=64):
    '''
    Function for duplicating an existing hdf5 file without a duplicate zarr file and saving it as a zarr file.
    Parallelized with configurable number of workers.
    '''
    print(f'Converting {hdf5_path} to zarr file in parallel')
    
    hdf5_file = h5py.File(hdf5_path, 'r')
    zarr_path = hdf5_path.replace('.hdf5', '.zarr')
    root = zarr.open(zarr_path, mode='w')

    def copy_dataset(item, zarr_node, key):
        ''' Copy HDF5 dataset to Zarr '''
        compression = item.compression
        compression_opts = item.compression_opts
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
    
def hdf5_to_zarr_parallel_with_progress(hdf5_path, max_workers=64):
    """
    Function for duplicating an existing hdf5 file without a duplicate zarr file
    and saving it as a zarr file. Parallelized with configurable number of workers,
    while showing a single tqdm progress bar. 
    """
    print(f'Converting {hdf5_path} to zarr file in parallel')
    
    # 1) Open source HDF5
    hdf5_file = h5py.File(hdf5_path, 'r')
    
    # 2) Prepare Zarr output path
    zarr_path = hdf5_path.replace('.hdf5', '.zarr')
    root = zarr.open(zarr_path, mode='w')

    # --A-- Count how many total keys we have (groups + datasets) to size the progress bar
    def count_keys(hdf_node):
        total = 0
        for _, item in hdf_node.items():
            total += 1
            if isinstance(item, h5py.Group):
                total += count_keys(item)
        return total

    total_keys = count_keys(hdf5_file)
    print(f"[INFO] Found {total_keys} total keys in {hdf5_path}")

    # --B-- We'll define the dataset copy function
    def copy_dataset(item, zarr_node, key):
        """ Copy HDF5 dataset to Zarr, reading everything in one go. """
        compression = item.compression
        compression_opts = item.compression_opts
        chunks = item.chunks if item.chunks else (1000,)
        zarr_node.create_dataset(
            key,
            data=item[:],
            chunks=chunks,
            dtype=item.dtype,
            compression=compression,
            compression_opts=compression_opts
        )
    
    # We'll capture the progress bar as an external variable
    pbar = None

    # --C-- Recursive function that processes groups/datasets in parallel
    def copy_node(hdf5_node, zarr_node):
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for key, item in hdf5_node.items():
                # Each key we encounter (whether group or dataset), we update progress by 1
                pbar.update(1)

                if isinstance(item, h5py.Group):
                    zarr_group = zarr_node.require_group(key)
                    # Recursively copy subgroups
                    futures.append(executor.submit(copy_node, item, zarr_group))
                elif isinstance(item, h5py.Dataset):
                    # Copy dataset in parallel
                    futures.append(executor.submit(copy_dataset, item, zarr_node, key))
            
            # Wait for all parallel tasks in this node
            concurrent.futures.wait(futures)

    # --D-- Create the single progress bar
    with tqdm(total=total_keys, desc="Copying Keys", ncols=100) as progress:
        pbar = progress  # let copy_node reference it
        copy_node(hdf5_file, root)

    # --E-- Done
    hdf5_file.close()
    print(f'Duplicated {hdf5_path} as zarr file {zarr_path}')
    
def hdf5_to_zarr_zip_parallel(hdf5_path, max_workers=64, chunk_len=100):
    """
    Converts an existing HDF5 file to a single-file .zarr.zip store, in parallel,
    but in a chunk-based manner to reduce memory usage. 
    
    1) We gather all dataset references.
    2) We copy each dataset chunk by chunk rather than item[:] 
       to avoid reading big arrays into memory all at once.
    3) We keep parallelism moderate (max_workers=4 by default).
    """

    print(f'Converting {hdf5_path} to zarr file in parallel')

    # 1) Open HDF5 source
    hdf5_file = h5py.File(hdf5_path, 'r')
    zarr_path = hdf5_path.replace('.hdf5', '.zarr.zip')

    # 2) Open/create the ZipStore
    store = ZipStore(zarr_path, mode='w')
    root = zarr.group(store, overwrite=True)

    # ----------------------------------------------------------
    # A) FIRST: SCAN the entire HDF5 to count how many keys we have
    # ----------------------------------------------------------
    def gather_items_count(hdf5_node):
        count = 0
        for key in hdf5_node.keys():
            count += 1
            item = hdf5_node[key]
            if isinstance(item, h5py.Group):
                count += gather_items_count(item)
        return count

    total_keys = gather_items_count(hdf5_file)
    print(f"Total keys (groups + datasets) in HDF5: {total_keys}")

    # ----------------------------------------------------------
    # B) GATHER: Recursively gather all HDF5 datasets, updating a TQDM bar
    # ----------------------------------------------------------
    items_to_copy = []

    def gather_items(hdf5_node, zarr_node, pbar):
        """
        Recursively walk HDF5 node, collecting references to datasets
        in `items_to_copy` while updating the progress bar.
        """
        for key in hdf5_node.keys():
            item = hdf5_node[key]
            pbar.update(1)
            if isinstance(item, h5py.Group):
                sub_group = zarr_node.require_group(key)
                gather_items(item, sub_group, pbar)
            elif isinstance(item, h5py.Dataset):
                items_to_copy.append((item, zarr_node, key))

    with tqdm(total=total_keys, desc="Gathering keys", ncols=100) as gather_pbar:
        gather_items(hdf5_file, root, gather_pbar)

    # ----------------------------------------------------------
    # C) COPY: chunk-based approach to reduce memory usage
    # ----------------------------------------------------------
    def copy_dataset_in_chunks(h5_dataset, zarr_node, key):
        """
        Copy an HDF5 dataset to Zarr chunk by chunk to avoid large memory usage.
        """
        shape = h5_dataset.shape
        dtype = h5_dataset.dtype
        # If user-specified compression is available, use it
        compression = h5_dataset.compression
        compression_opts = h5_dataset.compression_opts

        # Use existing chunk info or fallback
        chunks = h5_dataset.chunks if h5_dataset.chunks else (chunk_len,)

        # Create the Zarr dataset first (empty)
        # We do not do `data=item[:]`; instead we initialize an empty dataset.
        z = zarr_node.create_dataset(
            name=key,
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            compression=compression,
            compression_opts=compression_opts,
        )

        # We copy in slices along the first dimension (or you can adapt).
        if len(shape) == 0:
            # scalar dataset
            z[...] = h5_dataset[...]
            return
        elif shape[0] == 0:
            # empty dataset
            return

        # chunk copy
        chunk_size = chunk_len  
        total_len = shape[0]
        # For example, if shape = (10000, 128, 128, 3),
        # we iterate over dimension 0 in increments of chunk_size
        start = 0
        while start < total_len:
            end = min(start + chunk_size, total_len)
            # read only that slice from HDF5
            data_slice = h5_dataset[start:end]
            # write it to Zarr
            z[start:end] = data_slice
            start = end

    # Now we do the parallel copy
    with tqdm(total=len(items_to_copy), desc="Copying datasets", ncols=100) as copy_pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for (h5_ds, zarr_node, key) in items_to_copy:
                fut = executor.submit(copy_dataset_in_chunks, h5_ds, zarr_node, key)
                futures.append(fut)

            for _ in concurrent.futures.as_completed(futures):
                copy_pbar.update(1)

    # Close everything
    hdf5_file.close()
    store.close()
    print(f'Duplicated {hdf5_path} as zarr.zip file {zarr_path}')
    
def preprocess_data_parallel(root, camera, R, max_workers=4, batch_size=500, chunk_size=100):
    """
    Process demos in parallel, converting {camera}_image + {camera}_depth into
    {camera}_maskdepth, plus related camera-space transformations.
    Includes a TQDM progress bar. We now do chunk-based reading to reduce memory usage.
    """

    def process_demo(demo_key):
        obs_group = root['data'][demo_key]['obs']

        # If {camera}_maskdepth already exists, skip this demo
        maskdepth_key = f'{camera}_maskdepth'
        if maskdepth_key in obs_group:
            # Return None to indicate "already processed"
            return None

        # Access HDF5/Zarr arrays (but do NOT read them fully via [:])
        images_dset = obs_group[f'{camera}_image']      # shape ~ (trajectory_length, 128, 128, 3)
        depth_dset = obs_group[f'{camera}_depth']       # shape ~ (trajectory_length, 128, 128, 1)
        rgbd_dset  = obs_group[f'{camera}_rgbd']        # shape ~ (trajectory_length, 128, 128, 4)

        length = images_dset.shape[0]   # e.g. trajectory_length

        # Also read eef positions in smaller slices
        global_eef_dset = obs_group['robot0_eef_pos']   # shape ~ (trajectory_length, 3)

        # We'll accumulate final results across all chunks:
        #   maskdepth_array, camera_positions, disentangled_positions, rgbd_cropped
        # But we won't store them at full size in memory. Instead, we store them
        # incrementally. For demonstration, I'll store them as lists, then np.concatenate.
        # Another approach: directly write to a new dataset chunk by chunk.

        mask_list    = []
        campos_list  = []
        disent_list  = []
        rgbdcrop_list= []

        # For each chunk of frames
        for start in range(0, length, chunk_size):
            end = min(start + chunk_size, length)

            # 1) Read chunk of images
            images_chunk = images_dset[start:end]        # shape ~ (chunk, 128, 128, 3)
            depth_chunk  = depth_dset[start:end].squeeze(-1)  # shape ~ (chunk, 128, 128)
            rgbd_chunk   = rgbd_dset[start:end]          # shape ~ (chunk, 128, 128, 4)

            # 2) Convert chunk to HSV + create mask
            hsv_chunk = np.stack([
                cv2.cvtColor(img, cv2.COLOR_RGB2HSV) for img in images_chunk
            ])
            # green_lower, green_upper = np.array([40, 40, 90]), np.array([80, 255, 255])
            green_lower, green_upper = np.array([45, 80, 90]), np.array([75, 255, 255])
            cyan_lower,  cyan_upper  = np.array([80, 40, 100]), np.array([100, 255, 255])

            green_mask = ((hsv_chunk >= green_lower) & (hsv_chunk <= green_upper)).all(axis=-1).astype(np.uint8)*255
            cyan_mask  = ((hsv_chunk >= cyan_lower)  & (hsv_chunk <= cyan_upper)).all(axis=-1).astype(np.uint8)*255
            combined_mask = np.bitwise_or(green_mask, cyan_mask)

            # 3) Build maskdepth for this chunk
            chunk_len = images_chunk.shape[0]
            h, w = images_chunk.shape[1], images_chunk.shape[2]

            maskdepth_chunk = np.zeros((chunk_len, h, w, 3), dtype=np.uint8)
            maskdepth_chunk[..., 0] = green_mask
            maskdepth_chunk[..., 1] = cyan_mask
            maskdepth_chunk[..., 2] = np.where(combined_mask, depth_chunk, 0)

            # 4) EEF positions chunk
            global_eef_chunk = global_eef_dset[start:end]  # shape ~ (chunk, 3)
            # Convert to homogeneous
            ones = np.ones((global_eef_chunk.shape[0], 1), dtype=global_eef_chunk.dtype)
            hom_pos = np.hstack((global_eef_chunk, ones))       # (chunk, 4)
            # Apply extrinsic
            cam_pos_hom = (R @ hom_pos.T).T                     # (chunk, 4)
            cam_pos     = cam_pos_hom[:, :3]
            # Disentangled
            x, y, z = cam_pos[:, 0], cam_pos[:, 1], cam_pos[:, 2]
            disent_chunk = np.vstack((x/z, y/z, np.log(z))).T

            # 5) boundingRect for each frame in chunk
            bbox_coords = np.array([
                cv2.boundingRect(combined_mask[i]) for i in range(chunk_len)
            ])
            bbox_masks = np.zeros((chunk_len, h, w), dtype=np.uint8)
            for i, (x_, y_, w_, h_) in enumerate(bbox_coords):
                bbox_masks[i, y_:y_+h_, x_:x_+w_] = 1

            # expand + mask the rgbd chunk
            bbox_masks_expanded = bbox_masks[..., np.newaxis]
            rgbd_cropped_chunk  = rgbd_chunk * bbox_masks_expanded

            # Accumulate partial results in lists
            mask_list.append(maskdepth_chunk)
            campos_list.append(cam_pos)
            disent_list.append(disent_chunk)
            rgbdcrop_list.append(rgbd_cropped_chunk)

        # After finishing all chunks, we can concat
        maskdepth_array       = np.concatenate(mask_list, axis=0)
        camera_positions      = np.concatenate(campos_list, axis=0)
        disentangled_positions= np.concatenate(disent_list, axis=0)
        rgbd_cropped          = np.concatenate(rgbdcrop_list, axis=0)

        return (demo_key, maskdepth_array, camera_positions, disentangled_positions, rgbd_cropped)

    demo_keys = list(root['data'].keys())
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(demo_keys), desc=f"Preprocessing {camera}", ncols=100) as pbar:
            for demo_output in executor.map(process_demo, demo_keys):
                pbar.update(1)
                if demo_output is not None:
                    results.append(demo_output)

                if len(results) >= batch_size:
                    _write_batch_to_zarr(root, camera, results)
                    results.clear()

        if results:
            _write_batch_to_zarr(root, camera, results)


def _write_batch_to_zarr(root, camera, batch_results):
    """
    Writes a batch of results to Zarr. 
    This version avoids "deleting" the dataset if it exists.
    We simply create it if it's missing (which the logic above ensures).
    """
    for demo_key, maskdepth_array, camera_positions, disentangled_positions, rgbd_cropped in batch_results:
        obs_group = root['data'][demo_key]['obs']

        # Create maskdepth dataset if it doesn't exist
        maskdepth_key = f'{camera}_maskdepth'
        if maskdepth_key not in obs_group:
            obs_group.create_dataset(
                maskdepth_key, data=maskdepth_array,
                shape=maskdepth_array.shape, dtype=maskdepth_array.dtype
            )

        # EEF pos in camera coordinates
        eef_pos_key = f'robot0_eef_pos_{camera}'
        if eef_pos_key not in obs_group:
            obs_group.create_dataset(
                eef_pos_key, data=camera_positions,
                shape=camera_positions.shape, dtype=camera_positions.dtype
            )

        # EEF pos disentangled
        eef_pos_disentangled_key = f'robot0_eef_pos_{camera}_disentangled'
        if eef_pos_disentangled_key not in obs_group:
            obs_group.create_dataset(
                eef_pos_disentangled_key, data=disentangled_positions,
                shape=disentangled_positions.shape, dtype=disentangled_positions.dtype
            )

        # {camera}_rgbdcrop
        rgbdcrop_key = f'{camera}_rgbdcrop'
        if rgbdcrop_key not in obs_group:
            obs_group.create_dataset(
                rgbdcrop_key, data=rgbd_cropped,
                shape=rgbd_cropped.shape, dtype=rgbd_cropped.dtype
            )
            
########################
# 1) HDF5 -> DirectoryStore
########################
def hdf5_to_directorystore_chunked(hdf5_path, out_dir, max_workers=64, chunk_len=1000):
    """
    Reads a large HDF5 file and writes it as a Zarr DirectoryStore in a chunk-based,
    minimal-memory manner.
    
    :param hdf5_path: path to source .hdf5 file
    :param out_dir: directory for the resulting .zarr (DirectoryStore)
    :param max_workers: concurrency (set to 1 or 2 if memory is very tight)
    :param chunk_len: how many slices along dimension 0 to copy at once
    """
    import h5py
    import zarr
    from zarr import DirectoryStore
    import concurrent.futures
    from tqdm import tqdm
    import os

    print(f"[INFO] Converting HDF5 -> DirectoryStore:\n  {hdf5_path} -> {out_dir}")

    # Open source HDF5
    hdf5_file = h5py.File(hdf5_path, 'r')
    # Create or overwrite DirectoryStore
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    store = DirectoryStore(out_dir)
    root = zarr.group(store, overwrite=True)

    # Step A: Gather all datasets
    items_to_copy = []

    def gather_items(hdf5_node, zarr_node):
        for key, item in hdf5_node.items():
            if isinstance(item, h5py.Group):
                sub_group = zarr_node.require_group(key)
                gather_items(item, sub_group)
            elif isinstance(item, h5py.Dataset):
                items_to_copy.append((item, zarr_node, key))

    gather_items(hdf5_file, root)
    print(f"[INFO] Found {len(items_to_copy)} datasets to copy from HDF5.")

    # Step B: chunk-based copy for one dataset
    def copy_dataset_in_chunks(h5_dset, zarr_node, key):
        shape = h5_dset.shape
        dtype = h5_dset.dtype
        comp = h5_dset.compression
        comp_opts = h5_dset.compression_opts
        chunks = h5_dset.chunks or (chunk_len,)

        z = zarr_node.create_dataset(
            name=key,
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            compression=comp,
            compression_opts=comp_opts
        )

        if not shape or len(shape) == 0:
            # scalar
            z[...] = h5_dset[...]
            return
        if shape[0] == 0:
            return

        total_len = shape[0]
        start = 0
        while start < total_len:
            end = min(start + chunk_len, total_len)
            data_slice = h5_dset[start:end]
            z[start:end] = data_slice
            start = end

    # Step C: Parallel (or single-thread) copying
    with tqdm(total=len(items_to_copy), desc="Copying HDF5->DirStore") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for (h5_dset, znode, key) in items_to_copy:
                f = executor.submit(copy_dataset_in_chunks, h5_dset, znode, key)
                futures.append(f)

            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)

    hdf5_file.close()
    store.close()
    print(f"[INFO] Done writing to {out_dir}")


########################
# 2) DirectoryStore -> .zarr.zip
########################
def directorystore_to_zarr_zip(src_dir, out_zip):
    """
    After you've preprocessed the data in the DirectoryStore,
    convert it to a single .zarr.zip file for final distribution.
    """
    import zarr
    from zarr import ZipStore, DirectoryStore
    import os

    print(f"[INFO] Converting DirStore -> .zarr.zip:\n  {src_dir} -> {out_zip}")
    src = DirectoryStore(src_dir)
    dst = ZipStore(out_zip, mode='w')
    zarr.copy_store(src, dst)
    dst.close()
    print(f"[INFO] Created {out_zip} from {src_dir}")

def preprocess_maskdepth_data_parallel(root, camera, max_workers=8, batch_size=500):
    def process_demo(demo_key):
        print(f"Processing {demo_key} into {camera}_maskdepth")

        # Get the camera images and depth images
        images = root['data'][demo_key]['obs'][f'{camera}_image'][:]  # Shape: (trajectory_length, 128, 128, 3)
        depth_images = root['data'][demo_key]['obs'][f'{camera}_depth'][:]  # Shape: (trajectory_length, 128, 128, 1)
        depth_images = depth_images.squeeze(-1)

        # Convert all frames to HSV
        hsv_images = np.stack([cv2.cvtColor(img, cv2.COLOR_RGB2HSV) for img in images])

        # Define color ranges in HSV for green and cyan
        green_lower, green_upper = np.array([40, 40, 90]), np.array([80, 255, 255])
        cyan_lower, cyan_upper = np.array([80, 40, 100]), np.array([100, 255, 255])

        # Create masks for green and cyan
        green_mask = ((hsv_images >= green_lower) & (hsv_images <= green_upper)).all(axis=-1).astype(np.uint8) * 255
        cyan_mask = ((hsv_images >= cyan_lower) & (hsv_images <= cyan_upper)).all(axis=-1).astype(np.uint8) * 255

        # Union of green and cyan masks
        combined_mask = np.bitwise_or(green_mask, cyan_mask)

        # Create the mask-depth array by stacking green, cyan, and masked depth
        maskdepth_array = np.zeros((images.shape[0], images.shape[1], images.shape[2], 3), dtype=np.uint8)
        maskdepth_array[..., 0] = green_mask
        maskdepth_array[..., 1] = cyan_mask
        maskdepth_array[..., 2] = np.where(combined_mask, depth_images, 0)

        return demo_key, maskdepth_array

    # Process demos in parallel and write results in batches
    demo_keys = list(root['data'].keys())
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = []
        for demo_key, maskdepth_array in executor.map(process_demo, demo_keys):
            results.append((demo_key, maskdepth_array))

            # Write to Zarr every `batch_size` demos to prevent memory overflow
            if len(results) >= batch_size:
                _write_batch_to_zarr_(root, camera, results)
                results.clear()  # Clear the results list after writing to free up memory

        # Write any remaining results after processing all demos
        if results:
            _write_batch_to_zarr_(root, camera, results)

def _write_batch_to_zarr_(root, camera, batch_results):
    """Helper function to write a batch of results to Zarr."""
    for demo_key, maskdepth_array in batch_results:
        if f'{camera}_maskdepth' in root['data'][demo_key]['obs']:
            del root['data'][demo_key]['obs'][f'{camera}_maskdepth']  # Remove existing data if any
        root['data'][demo_key]['obs'].create_dataset(
            f'{camera}_maskdepth', data=maskdepth_array, shape=maskdepth_array.shape, dtype=maskdepth_array.dtype, overwrite=True
        )
        print(f"Saved {camera}_maskdepth for {demo_key}")

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
    
    
def save_maskdepth_visualization(original_image, maskdepth_array, save_dir="debug", filename="maskdepth.png"):
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    if isinstance(maskdepth_array, torch.Tensor) and list(maskdepth_array.size()) == [3, 128, 128]:
        maskdepth_array = maskdepth_array.permute(1, 2, 0).numpy()
    
    # Extract the three channels from the maskdepth array
    green_mask = maskdepth_array[..., 0]  # First channel (0s and 1s)
    cyan_mask = maskdepth_array[..., 1]   # Second channel (0s and 1s)
    depth_mask = maskdepth_array[..., 2]  # Third channel (0 to 255)
    
    # Convert the binary masks to be visible (0s and 1s to 0s and 255s)
    if green_mask.max() == 1.0:
        green_mask = (green_mask * 255).astype(np.uint8)
        cyan_mask = (cyan_mask * 255).astype(np.uint8)
        depth_mask = (depth_mask * 255).astype(np.uint8)
    
    # Set up the figure for displaying the images side by side
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    
    # Display the original image
    if isinstance(original_image, np.ndarray):
        axs[0].imshow(original_image)
        axs[0].set_title("Original Image")
        axs[0].axis("off")
    
    # Display the first channel (green mask) in black and white
    axs[1].imshow(green_mask, cmap="gray")
    axs[1].set_title("Green Mask (Channel 1)")
    axs[1].axis("off")
    
    # Display the second channel (cyan mask) in black and white
    axs[2].imshow(cyan_mask, cmap="gray")
    axs[2].set_title("Cyan Mask (Channel 2)")
    axs[2].axis("off")
    
    # Display the third channel (depth mask) in black and white
    axs[3].imshow(depth_mask, cmap="gray", vmin=0, vmax=255)
    axs[3].set_title("Depth Mask (Channel 3)")
    axs[3].axis("off")
    
    # Save the figure as an image
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Maskdepth visualization saved at {save_path}")
    
    
def quaternion_inverse(q):
        '''Assumes q is a normalized quaternion (w, x, y, z) and returns its inverse.'''
        return np.array([q[0], -q[1], -q[2], -q[3]])

def quaternion_multiply(q1, q2):
    """Multiplies two quaternions q1 and q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
        w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    ])

def quaternion_difference(q1, q2):
    """Calculates the difference between two quaternions q1 and q2."""
    q1_inv = quaternion_inverse(q1)
    return quaternion_multiply(q2, q1_inv)

def get_point_in_camera_frame(point_3D, R):
    point_3D_homogeneous = np.array([point_3D[0], point_3D[1], point_3D[2], 1])

    # Transform the point from world coordinates to camera coordinates
    point_camera = R @ point_3D_homogeneous  # Resulting shape (4,)

    # Extract x, y, and z coordinates in camera space
    y_cam, x_cam, z_cam, _ = point_camera
    
    return np.array([x_cam, y_cam, z_cam])

def project_point(K, R, point_3D):
    """
    Projects a 3D point in world coordinates onto the 2D image plane.
    
    Parameters:
    - K: 3x3 intrinsic matrix
    - R: 4x4 extrinsic matrix (contains both rotation and translation)
    - point_3D: 3D point in world coordinates, given as (X, Y, Z)

    Returns:
    - A tuple (u, v) representing the 2D pixel coordinates.
    """
    # Convert the 3D point to homogeneous coordinates
    point_3D_homogeneous = np.array([point_3D[0], point_3D[1], point_3D[2], 1])

    # Transform the point from world coordinates to camera coordinates
    point_camera = R @ point_3D_homogeneous  # Resulting shape (4,)

    # Extract x, y, and z coordinates in camera space
    x_cam, y_cam, z_cam, _ = point_camera
    if z_cam == 0:  # Avoid division by zero
        raise ValueError("Point is located on the camera plane (z=0), projection undefined.")

    # Apply the intrinsic matrix to the normalized camera coordinates
    pixel_coords_homogeneous = K @ np.array([x_cam, y_cam, z_cam])

    # Normalize to get pixel coordinates in 2D
    u = pixel_coords_homogeneous[0] / pixel_coords_homogeneous[2]
    v = pixel_coords_homogeneous[1] / pixel_coords_homogeneous[2]

    return (u, v)

def visualize_visible_points(K, R, x_range, y_range, z_range):
    fig, ax = plt.subplots()
    ax.set_xlim(0, K[0, 2] * 2)
    ax.set_ylim(0, K[1, 2] * 2)
    ax.invert_yaxis()

    for x in np.linspace(x_range[0], x_range[1], 10):
        for y in np.linspace(y_range[0], y_range[1], 10):
            for z in np.linspace(z_range[0], z_range[1], 10):
                u, v = project_point(K, R, np.array([x, y, z]))
                ax.plot(u, v, 'ro')

    plt.show()

def draw_and_save_point(image, point):
    point_radius = 2
    point_color = (0, 255, 0)  # Blue color in RGB
    if image.max() == 1.0:
        image = (image * 255).astype(np.uint8)
        
    point = (int(np.round(point[0])), int(np.round(point[1])))

    # Draw the point on the image using OpenCV
    cv2.circle(image, point, point_radius, point_color, -1)  # -1 fills the circle

    # Save the image in the debug directory as 'gripper_position.png'
    cv2.imwrite("debug/gripper_position.png", image)
    
    
def quaternion_to_rotation_matrix(Q):
    """
    Converts a quaternion to a 3x3 rotation matrix.
    
    Parameters:
    - quaternion: A list or array with four elements [w, x, y, z] where
                  w is the scalar part and (x, y, z) are the vector parts.

    Returns:
    - A 3x3 rotation matrix as a NumPy array.
    """
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix

def pose_inv(pose):
    """
    From /robosuite/utils/transform_utils.py
    
    Computes the inverse of a homogeneous matrix corresponding to the pose of some
    frame B in frame A. The inverse is the pose of frame A in frame B.

    Args:
        pose (np.array): 4x4 matrix for the pose to inverse

    Returns:
        np.array: 4x4 matrix for the inverse pose
    """

    # Note, the inverse of a pose matrix is the following
    # [R t; 0 1]^-1 = [R.T -R.T*t; 0 1]

    # Intuitively, this makes sense.
    # The original pose matrix translates by t, then rotates by R.
    # We just invert the rotation by applying R-1 = R.T, and also translate back.
    # Since we apply translation first before rotation, we need to translate by
    # -t in the original frame, which is -R-1*t in the new frame, and then rotate back by
    # R-1 to align the axis again.

    pose_inv = np.zeros((4, 4))
    pose_inv[:3, :3] = pose[:3, :3].T
    pose_inv[:3, 3] = -pose_inv[:3, :3].dot(pose[:3, 3])
    pose_inv[3, 3] = 1.0
    return pose_inv

# Load camera matrices using importlib.resources
with importlib.resources.path('action_extractor.utils', 'frontview_matrices.npz') as frontview_path:
    frontview_matrices = np.load(frontview_path)
    frontview_K = frontview_matrices['K']  # Intrinsics
    frontview_R = pose_inv(frontview_matrices['R'])  # Extrinsics

with importlib.resources.path('action_extractor.utils', 'sideview_matrices.npz') as sideview_path:
    sideview_matrices = np.load(sideview_path)
    sideview_K = sideview_matrices['K']
    sideview_R = pose_inv(sideview_matrices['R'])

with importlib.resources.path('action_extractor.utils', 'agentview_matrices.npz') as agentview_path:
    agentview_matrices = np.load(agentview_path)
    agentview_K = agentview_matrices['K']
    agentview_R = pose_inv(agentview_matrices['R'])

with importlib.resources.path('action_extractor.utils', 'sideagentview_matrices.npz') as sideagentview_path:
    sideagentview_matrices = np.load(sideagentview_path)
    sideagentview_K = sideagentview_matrices['K']
    sideagentview_R = pose_inv(sideagentview_matrices['R'])
    
camera_extrinsics = {
    'frontview': frontview_R,
    'sideview': sideview_R,
    'agentview': agentview_R,
    'sideagentview': sideagentview_R
}


def save_image_to_debug(image, filename="image.png"):
    """
    Saves an RGB image to the debug directory using matplotlib.

    Parameters:
    - image (np.ndarray): A (128, 128, 3) RGB image.
    - filename (str): The name of the file to save. Default is "image.png".
    """
    # Ensure the image has the correct shape and data type
    if image.shape != (128, 128, 3) or image.dtype != np.uint8:
        raise ValueError("Image must be a (128, 128, 3) array with dtype=np.uint8.")

    # Ensure the debug directory exists
    os.makedirs("debug", exist_ok=True)

    # Save the image in the debug directory
    save_path = os.path.join("debug", filename)
    plt.imsave(save_path, image)
    print(f"Image saved to {save_path}")
    
    
def get_visible_xyz_range(extrinsics, intrinsics, z_range=(0.1, 10)):
    """
    Calculates the range of x, y, z positions in the global frame that would be visible within the camera's frame.

    Parameters:
    - extrinsics (numpy.ndarray): 4x4 camera extrinsics matrix (world to camera transformation).
    - intrinsics (numpy.ndarray): 3x3 camera intrinsics matrix.
    - z_range (tuple): The minimum and maximum depth (z) values to consider in the global frame.

    Returns:
    - x_range (tuple): The minimum and maximum x values in the global frame.
    - y_range (tuple): The minimum and maximum y values in the global frame.
    - z_range (tuple): The minimum and maximum z values in the global frame.
    """
    # Define image corners in normalized pixel coordinates (top-left, top-right, bottom-right, bottom-left)
    img_corners = np.array([
        [0, 0],  # top-left
        [intrinsics[0, 2] * 2, 0],  # top-right (2 * cx)
        [intrinsics[0, 2] * 2, intrinsics[1, 2] * 2],  # bottom-right (2 * cx, 2 * cy)
        [0, intrinsics[1, 2] * 2]  # bottom-left (0, 2 * cy)
    ])

    # Convert image plane corners to camera coordinates (homogeneous coordinates)
    camera_corners = []
    for u, v in img_corners:
        # Calculate the direction vector in camera coordinates
        pixel_vec = np.linalg.inv(intrinsics) @ np.array([u, v, 1])
        pixel_vec /= np.linalg.norm(pixel_vec)

        # Project to z-range in camera coordinates
        for z in z_range:
            camera_corners.append(pixel_vec * z)

    # Convert camera frame corners to the world frame
    camera_corners = np.array(camera_corners)
    global_corners = np.linalg.inv(extrinsics) @ np.hstack((camera_corners, np.ones((camera_corners.shape[0], 1)))).T

    # Extract x, y, z ranges from the global frame corners
    x_min, x_max = np.min(global_corners[0]), np.max(global_corners[0])
    y_min, y_max = np.min(global_corners[1]), np.max(global_corners[1])
    z_min, z_max = np.min(global_corners[2]), np.max(global_corners[2])

    return (x_min, x_max), (y_min, y_max), (z_min, z_max)