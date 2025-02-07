import h5py

hdf5_path = "/home/yilong/Documents/policy_data/square_d0/raw/square_d0.hdf5"

with h5py.File(hdf5_path, "r") as f:
    # Access the "data" group
    data_group = f["data"]
    # Count how many demos (keys) are inside data_group
    num_demos = len(data_group.keys())
    print(f"Number of demos in 'data': {num_demos}")
    
