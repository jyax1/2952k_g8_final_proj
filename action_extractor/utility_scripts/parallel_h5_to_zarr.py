#!/usr/bin/env python3

"""
Copy a large HDF5 file into a Zarr store in parallel, focusing on:
 - 'data' top-level group with many subitems,
 - optional 'mask' group,
 - chunk-based reading/writing to handle extremely large datasets.

Preserved Features:
 - Parallel subitem copying using multiprocessing.
 - Error printing if a worker fails.
 - TQDM progress bar for overall subitem progress.

Usage:
    python parallel_h5_to_zarr.py /path/to/input.h5 [num_workers]

Creates `/path/to/input.zarr` in the same directory.
"""

import os
import sys
import traceback
import h5py
import zarr
import numpy as np
from numcodecs import GZip
import multiprocessing
from tqdm import tqdm


def copy_dataset_chunked(h5_dataset, zarr_group, dataset_name, compressor=None):
    """
    Copy an HDF5 dataset to a Zarr dataset, chunk by chunk.

    - We create a Zarr dataset with the same shape, dtype, and (if available) chunk layout.
    - Then we iterate over each chunk (slicing the HDF5 dataset) and write it to Zarr.

    This prevents loading the entire dataset into RAM at once.
    """
    shape = h5_dataset.shape
    dtype = h5_dataset.dtype
    h5_chunks = h5_dataset.chunks  # might be None
    rank = len(shape)

    # If the HDF5 dataset has no chunk layout, define a chunk size ourselves:
    # e.g., read in slices of ~64MB if possible, or some heuristic.
    if h5_chunks is None:
        # Basic approach: pick a chunk size that is the same in all dims except
        # the first dimension chunked to 1, for instance, or any strategy that
        # balances memory usage. We'll do a naive approach:
        # (You may want a more sophisticated chunking logic for extremely high dims.)
        chunk_size = []
        for dim_size in shape:
            # e.g., chunk each dimension to min(dim_size, 64) or something
            chunk_size.append(min(dim_size, 64))
        h5_chunks = tuple(chunk_size)

    # Create the Zarr dataset. We'll use the same chunk shape as HDF5 if available.
    zarr_dataset = zarr_group.create_dataset(
        name=dataset_name,
        shape=shape,
        dtype=dtype,
        chunks=h5_chunks,
        compressor=compressor,
    )

    # We'll iterate over the chunk grid. If shape = (D0, D1, ...), and chunk = (C0, C1, ...),
    # then number of chunks along dimension i is ceil(Di / Ci).
    def chunk_slices(shape, chunks):
        """Generate slice tuples for each chunk in a dataset of 'shape' with 'chunks'."""
        ranges = []
        for dim_size, chunk_size in zip(shape, chunks):
            # Create a list of start indices for this dimension
            indices = list(range(0, dim_size, chunk_size))
            ranges.append(indices)

        # Now build slices
        # e.g. for 2D, shape=(100,200), chunks=(32,32), we have
        # dimension 0 => [0, 32, 64, 96]
        # dimension 1 => [0, 32, 64, 96, 128, 160, 192]
        # We'll yield a slice for each combination.
        # E.g., for dim0 start=32, dim1 start=64 => slice(32,64), slice(64,96).
        import math
        if rank == 0:
            # Scalar dataset
            yield (slice(None),)
        else:
            # multi-dim
            from itertools import product
            dim_starts = [range_list for range_list in ranges]
            for coords in product(*dim_starts):
                s = []
                for d, start in enumerate(coords):
                    end = min(start + chunks[d], shape[d])
                    s.append(slice(start, end))
                yield tuple(s)

    for s in chunk_slices(shape, h5_chunks):
        # Read from HDF5
        arr = h5_dataset[s]
        # Write to Zarr
        zarr_dataset[s] = arr


def copy_h5_group_to_zarr_group(h5_group, zarr_group):
    """
    Recursively copy items from an h5py.Group to a zarr.Group (single-process),
    using chunk-based copying for large datasets.
    """
    for key, item in h5_group.items():
        if isinstance(item, h5py.Group):
            sub_zgroup = zarr_group.create_group(key)
            copy_h5_group_to_zarr_group(item, sub_zgroup)
        elif isinstance(item, h5py.Dataset):
            # Attempt to replicate gzip
            compressor = None
            if item.compression == 'gzip':
                level = item.compression_opts or 4
                compressor = GZip(level=level)

            copy_dataset_chunked(item, zarr_group, key, compressor=compressor)
        else:
            print(f"[WARN] Skipping unknown item '{key}' (type: {type(item)})")


def copy_h5_subitem(args):
    """
    Worker function that copies one subitem (group or dataset) under /data.
    
    We open the HDF5 file in read-only in this process, copy chunk by chunk.

    Args: (h5_filepath, zarr_path, subitem_name)

    Returns: (subitem_name, error_or_None)
    """
    h5_filepath, zarr_path, subitem_name = args
    try:
        with h5py.File(h5_filepath, "r") as h5f:
            root_z = zarr.open(zarr_path, mode="a")  # append
            data_group_h5 = h5f["data"]
            data_group_zarr = root_z.require_group("data")

            h5_item = data_group_h5[subitem_name]
            if isinstance(h5_item, h5py.Group):
                z_subgroup = data_group_zarr.require_group(subitem_name)
                copy_h5_group_to_zarr_group(h5_item, z_subgroup)
            elif isinstance(h5_item, h5py.Dataset):
                # chunk-based copy
                compressor = None
                if h5_item.compression == 'gzip':
                    level = h5_item.compression_opts or 4
                    compressor = GZip(level=level)
                copy_dataset_chunked(h5_item, data_group_zarr, subitem_name, compressor=compressor)
            else:
                print(f"[WARN] '/data/{subitem_name}' is neither group nor dataset.")

        return (subitem_name, None)

    except Exception:
        tb_str = traceback.format_exc()
        return (subitem_name, tb_str)


def copy_mask_group(h5_filepath, zarr_path):
    """
    Copy the 'mask' group in single-process mode (if it exists), chunk-based.
    """
    with h5py.File(h5_filepath, "r") as h5f:
        if "mask" not in h5f:
            print("[INFO] No 'mask' group found in HDF5.")
            return
        mask_h5 = h5f["mask"]

        root_z = zarr.open(zarr_path, mode="a")
        mask_z = root_z.require_group("mask")

        copy_h5_group_to_zarr_group(mask_h5, mask_z)

    print("[INFO] Copied 'mask' group successfully.")


def main(h5_filepath, num_workers=4):
    # Derive Zarr path in same directory
    dir_name, base_name = os.path.split(h5_filepath)
    base_no_ext, _ = os.path.splitext(base_name)
    zarr_name = base_no_ext + ".zarr"
    zarr_path = os.path.join(dir_name, zarr_name)

    print(f"\nHDF5 file: {h5_filepath}")
    print(f"Target Zarr: {zarr_path}")
    print(f"Using {num_workers} worker processes.\n")

    # Create fresh Zarr store
    zarr.open(zarr_path, mode="w")

    # Gather subitems from /data
    with h5py.File(h5_filepath, "r") as h5f:
        if "data" not in h5f:
            print("[ERROR] No 'data' group at top level. Exiting.")
            sys.exit(1)

        data_group = h5f["data"]
        subitems = list(data_group.keys())
        print(f"[INFO] Found {len(subitems)} subitems under '/data'.")

    # Parallel copy of subitems under /data
    tasks = [(h5_filepath, zarr_path, sub) for sub in subitems]

    from tqdm import tqdm
    print("[INFO] Starting parallel copy of /data subitems...")
    results = []
    with multiprocessing.Pool(processes=num_workers) as pool, tqdm(total=len(tasks)) as pbar:
        for res in pool.imap_unordered(copy_h5_subitem, tasks):
            results.append(res)
            pbar.update(1)

    # Print any errors
    for (subitem_name, err) in results:
        if err is not None:
            print(f"[ERROR] Copy of '/data/{subitem_name}' failed:\n{err}")

    # Single-process copy for '/mask' (if it exists)
    copy_mask_group(h5_filepath, zarr_path)

    print("\nAll done! Check above for any error messages.\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parallel_h5_to_zarr.py /path/to/input.h5 [num_workers]")
        sys.exit(1)

    h5_file = sys.argv[1]
    if not os.path.isfile(h5_file):
        print(f"Error: file '{h5_file}' not found.")
        sys.exit(1)

    workers = 64
    if len(sys.argv) >= 3:
        workers = int(sys.argv[2])

    main(h5_file, workers)
