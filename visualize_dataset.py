import os
import cv2
import numpy as np
import h5py

# Define the path to the HDF5 file and output directory
hdf5_file_path = '/home/yilong/Documents/ae_data/random_processing/obs_rel_color_smoothg_sideagent/organized_vSmoothGripper.hdf5'
output_dir = 'debug/smooth_gripper_agent'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Open the HDF5 file
with h5py.File(hdf5_file_path, 'r') as root:
    
    i = 0
    # Iterate over each demo
    for demo in root['data']:
        if i == 500:
            break
        
        # Retrieve the frontview_image for this demo
        frontview_images = root['data'][demo]['obs']['agentview_image'][:]
        
        # Get the number of frames (n), width, height, and channels
        n, height, width, channels = frontview_images.shape
        
        # Define the video codec and create a VideoWriter object
        video_path = os.path.join(output_dir, f'{demo}.avi')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = 30  # You can adjust the frames per second
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        # Loop over frames and write each frame to the video
        for frame in frontview_images:
            # Convert the frame from RGB (if needed) to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        # Release the VideoWriter
        out.release()
        
        i += 1

print("Videos saved successfully.")