import h5py

def print_demo_obs_info(hdf5_path):
    """
    Opens an HDF5 file at `hdf5_path`, navigates to root['data']['demo_0']['obs'],
    and prints each key's shape, dtype, and storage size in bytes.
    """
    with h5py.File(hdf5_path, 'r') as f:
        # Navigate to the desired group
        demo_obs_group = f['data']['demo_0']['obs']
        
        # For each key, print shape, dtype, and storage size
        for key in demo_obs_group.keys():
            dset = demo_obs_group[key]
            shape = dset.shape
            dtype = dset.dtype
            size_in_bytes = dset.id.get_storage_size()
            print(f"Key: {key}, Shape: {shape}, DType: {dtype}, Size (bytes): {size_in_bytes}")

if __name__ == "__main__":
    # Replace with the path to your .hdf5 file
    file_path = "/home/yilong/Documents/policy_data/square_d0/raw/first100/square_d0_obs_first100.hdf5"
    print_demo_obs_info(file_path)