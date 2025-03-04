#!/usr/bin/env python3
import argparse
import os
import h5py
import imageio
from tqdm import tqdm

def visualize_dataset(hdf5_path: str, out_dir: str, fps: int = 20):
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    with h5py.File(hdf5_path, 'r') as f:
        # Assumes demos are stored under the group "data"
        data_group = f['data']
        demo_names = list(data_group.keys())
        print(f"Found {len(demo_names)} demos.")

        for demo_name in tqdm(demo_names, desc="Processing demos"):
            demo_group = data_group[demo_name]
            obs_group = demo_group['obs']
            # Identify camera keys (we assume keys that end with '_image' are cameras)
            camera_names = [k for k in obs_group.keys() if k.endswith('_image')]
            if not camera_names:
                print(f"No camera images found in demo {demo_name}")
                continue
            
            # For each camera in this demo, read frames and write video.
            for cam in camera_names:
                frames = obs_group[cam][:]
                # Optional: If your images are stored as floats (0-1), you may need to scale to 0-255
                # frames = (frames * 255).astype("uint8")
                video_filename = f"{demo_name}_{cam}.mp4"
                video_path = os.path.join(out_dir, video_filename)
                
                # Write video using imageio
                with imageio.get_writer(video_path, fps=fps) as writer:
                    for frame in frames:
                        writer.append_data(frame)
                print(f"Saved video: {video_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Visualize demos from an HDF5 observation dataset and save as videos."
    )
    parser.add_argument(
        "hdf5_file",
        type=str,
        help="Path to the HDF5 dataset file (e.g., /home/yilong/Documents/robo/robomimic_3d/datasets/square/ph/square_d0_obs_orig_black.hdf5)"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="visualizations/dataset_vis",
        help="Directory to save the output videos."
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Frames per second for the output videos."
    )
    args = parser.parse_args()
    
    visualize_dataset(args.hdf5_file, args.out_dir, args.fps)

if __name__ == "__main__":
    main()