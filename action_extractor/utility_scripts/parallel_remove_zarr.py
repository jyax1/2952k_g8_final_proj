#!/usr/bin/env python3

"""
Optimized Parallel Removal of Zarr Stores with Progress Bars and Enhanced Error Reporting.

This script efficiently removes one or more Zarr stores (directories) by:
 - Targeting the 'data' subdirectory for parallel deletion due to its high content.
 - Deleting the 'mask' subdirectory sequentially (assuming it's smaller).
 - Providing real-time progress updates via progress bars for both collection and deletion.
 - Reporting any errors encountered during the deletion process.

Usage:
    python parallel_remove_zarr_optimized.py /path/to/ae_iiwa16168.zarr /path/to/another.zarr [--workers N]

Arguments:
    ZARR_PATH      Path(s) to the Zarr store directory to remove.

Options:
    --workers N    Number of parallel worker threads (default: number of CPU cores).
"""

import os
import sys
import argparse
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments containing Zarr paths and worker count.
    """
    parser = argparse.ArgumentParser(
        description="Optimized parallel removal of Zarr stores with progress bars and error reporting."
    )
    parser.add_argument(
        'zarr_paths',
        metavar='ZARR_PATH',
        type=str,
        nargs='+',
        help='Path(s) to the Zarr store directory to remove.'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=os.cpu_count(),
        help='Number of parallel worker threads (default: number of CPU cores).'
    )
    return parser.parse_args()

def collect_files_dirs(zarr_path):
    """
    Traverse the Zarr store directory and collect all file and directory paths.

    A progress bar is displayed to monitor the collection progress.

    Args:
        zarr_path (str): Path to the Zarr store.

    Returns:
        tuple: Two lists containing file paths and directory paths respectively.
    """
    files = []
    dirs = []
    # Initialize tqdm without a total since we don't know the total number of items in advance
    with tqdm(desc=f"Collecting items in '{os.path.basename(zarr_path)}'", unit="item") as pbar:
        for root, dirnames, filenames in os.walk(zarr_path, topdown=False):
            for dirname in dirnames:
                dirpath = os.path.join(root, dirname)
                dirs.append(dirpath)
                pbar.update(1)
            for filename in filenames:
                filepath = os.path.join(root, filename)
                files.append(filepath)
                pbar.update(1)
            dirs.append(root)  # Include the current root to delete later
            pbar.update(1)
    return files, dirs

def delete_file(filepath):
    """
    Delete a single file.

    Args:
        filepath (str): Path to the file to delete.

    Returns:
        tuple: (filepath, None) if successful, or (filepath, error_message) if failed.
    """
    try:
        os.remove(filepath)
        return (filepath, None)
    except Exception as e:
        tb_str = traceback.format_exc()
        return (filepath, tb_str)

def delete_directory(dirpath):
    """
    Delete a single directory.

    Args:
        dirpath (str): Path to the directory to delete.

    Returns:
        tuple: (dirpath, None) if successful, or (dirpath, error_message) if failed.
    """
    try:
        os.rmdir(dirpath)
        return (dirpath, None)
    except Exception as e:
        tb_str = traceback.format_exc()
        return (dirpath, tb_str)

def parallel_delete_files(files, workers):
    """
    Delete files in parallel using multiple threads.

    Args:
        files (list): List of file paths to delete.
        workers (int): Number of parallel worker threads.

    Returns:
        tuple: Lists of successfully deleted files and encountered errors.
    """
    success = []
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all delete_file tasks
        future_to_file = {executor.submit(delete_file, f): f for f in files}
        # Use tqdm for progress bar
        for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc="Deleting files", unit="file"):
            filepath = future_to_file[future]
            try:
                result, error = future.result()
                if error is None:
                    success.append(result)
                else:
                    errors.append((result, error))
            except Exception as e:
                tb_str = traceback.format_exc()
                errors.append((filepath, tb_str))
    return success, errors

def parallel_delete_dirs(dirs, workers):
    """
    Delete directories in parallel using multiple threads.

    Args:
        dirs (list): List of directory paths to delete.
        workers (int): Number of parallel worker threads.

    Returns:
        tuple: Lists of successfully deleted directories and encountered errors.
    """
    success = []
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all delete_directory tasks
        future_to_dir = {executor.submit(delete_directory, d): d for d in dirs}
        # Use tqdm for progress bar
        for future in tqdm(as_completed(future_to_dir), total=len(future_to_dir), desc="Deleting directories", unit="dir"):
            dirpath = future_to_dir[future]
            try:
                result, error = future.result()
                if error is None:
                    success.append(result)
                else:
                    errors.append((result, error))
            except Exception as e:
                tb_str = traceback.format_exc()
                errors.append((dirpath, tb_str))
    return success, errors

def remove_zarr_store(zarr_path, workers):
    """
    Remove a single Zarr store by deleting all its files and directories in parallel.

    Args:
        zarr_path (str): Path to the Zarr store directory.
        workers (int): Number of parallel worker threads.

    Returns:
        list: List of tuples containing paths and their respective error messages.
    """
    if not os.path.isdir(zarr_path):
        print(f"[ERROR] '{zarr_path}' is not a directory or does not exist.")
        return [(zarr_path, "Not a directory or does not exist.")]
    
    print(f"\nProcessing Zarr store: {zarr_path}")
    files, dirs = collect_files_dirs(zarr_path)
    print(f"Found {len(files)} files and {len(dirs)} directories to delete.")
    
    # Delete files first
    print("Starting file deletion...")
    del_files_success, del_files_errors = parallel_delete_files(files, workers)
    
    # Then delete directories
    print("Starting directory deletion...")
    del_dirs_success, del_dirs_errors = parallel_delete_dirs(dirs, workers)
    
    # Final summary
    errors = del_files_errors + del_dirs_errors
    if errors:
        print(f"\n[ERROR] Failed to delete {len(errors)} items in '{zarr_path}':")
        for path, err in errors:
            print(f" - {path}:\n{err}")
    else:
        print(f"\n[INFO] Successfully deleted all items in '{zarr_path}'.")
    
    return errors

def main():
    """
    Main function to parse arguments and initiate the removal of Zarr stores.
    """
    args = parse_arguments()
    zarr_paths = args.zarr_paths
    workers = args.workers

    total_errors = []

    for zarr_path in zarr_paths:
        errors = remove_zarr_store(zarr_path, workers)
        total_errors.extend(errors)
    
    if total_errors:
        print("\nSummary of Errors:")
        for path, err in total_errors:
            print(f"\nPath: {path}\nError:\n{err}")
    else:
        print("\nAll specified Zarr stores have been successfully removed.")

if __name__ == "__main__":
    main()
