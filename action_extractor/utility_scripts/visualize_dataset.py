import os
import h5py
from PIL import Image
import cv2
import numpy as np

# Define the path to the HDF5 file and output directory
hdf5_file_path = '/home/yilong/Documents/ae_data/random_processing/iiwa200/action_extraction_IIWA200.hdf5'
output_dir = 'debug/D_movement_iiwa200'
output_format = 'mp4'  # Choose 'mp4' or 'webp'

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
        frontview_images = root['data'][demo]['obs']['frontview_image'][:]
        
        if output_format == 'mp4':
            # Save as MP4 video
            # Get the number of frames, height, width, and channels
            n_frames, height, width, channels = frontview_images.shape
            
            # Define the video codec and create a VideoWriter object
            video_path = os.path.join(output_dir, f'{demo}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30  # Adjust the frames per second if needed
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            
            # Loop over frames and write each frame to the video
            for frame in frontview_images:
                # Convert the frame from RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            # Release the VideoWriter
            out.release()
        
        elif output_format == 'webp':
            # Save as animated WebP
            # Convert images to a list of PIL Images
            pil_images = [Image.fromarray(frame) for frame in frontview_images]
            
            # Save as animated WebP
            output_path = os.path.join(output_dir, f'{demo}.webp')
            pil_images[0].save(
                output_path, 
                format='WEBP', 
                save_all=True, 
                append_images=pil_images[1:], 
                duration=33,  # Duration between frames in milliseconds (30 FPS)
                loop=0        # 0 means loop indefinitely
            )
        else:
            print(f"Unsupported output format: {output_format}")
            break

        i += 1

print("Videos saved successfully.")