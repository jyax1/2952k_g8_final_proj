#####################################################################
# Add these lines BEFORE other Panda3D imports to silence logs:
#####################################################################
import logging

# Silence "Loading model ..." logs from your python code
logging.getLogger("action_extractor.megapose.action_identifier_megapose").setLevel(logging.WARNING)

import panda3d.core as p3d
# Suppress "Known pipe types", "Xlib extension missing", etc.
p3d.load_prc_file_data("", "notify-level-glxdisplay fatal\n")
p3d.load_prc_file_data("", "notify-level-x11display fatal\n")
p3d.load_prc_file_data("", "notify-level-gsg fatal\n")

#####################################################################
# Normal imports
#####################################################################
import os
import h5py
import numpy as np
import torch
from tqdm import tqdm
import imageio
import cv2
import zarr
from glob import glob
from einops import rearrange
from zarr import DirectoryStore, ZipStore
import shutil
import math

from action_extractor.action_identifier import load_action_identifier, VariationalEncoder
from action_extractor.utils.dataset_utils import (
    hdf5_to_zarr_parallel_with_progress,
    preprocess_data_parallel,
    directorystore_to_zarr_zip,
    frontview_K, frontview_R, sideview_K, sideview_R,
    agentview_K, agentview_R, sideagentview_K, sideagentview_R,
    pose_inv
)
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.file_utils import get_env_metadata_from_dataset
from robomimic.utils.env_utils import create_env_from_metadata

from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from pathlib import Path
from megapose.utils.logging import get_logger
logger = get_logger(__name__)

from megapose.utils.load_model import NAMED_MODELS, load_named_model
from action_extractor.utils.dataset_utils import (
    hdf5_to_zarr_parallel_with_progress,
    directorystore_to_zarr_zip,
)

from action_extractor.megapose.action_identifier_megapose import (
    make_object_dataset_from_folder,
)

from robosuite.utils.camera_utils import get_real_depth_map, get_camera_extrinsic_matrix, get_camera_intrinsic_matrix

#####################################################################
# Instead of from action_extractor.megapose.action_identifier_megapose 
# we STILL import the same run_inference_on_data, but we will pass 
# our pre-loaded model to it so it doesn't re-load. We'll define 
# a small wrapper or local approach for that.
#####################################################################
from action_extractor.megapose.action_identifier_megapose import make_object_dataset_from_folder

def combine_videos_quadrants(top_left_video_path, top_right_video_path, 
                           bottom_left_video_path, bottom_right_video_path, 
                           output_path):
    """Combines four videos into a single quadrant layout video."""
    import imageio

    top_left_reader = imageio.get_reader(top_left_video_path)
    top_right_reader = imageio.get_reader(top_right_video_path)
    bottom_left_reader = imageio.get_reader(bottom_left_video_path)
    bottom_right_reader = imageio.get_reader(bottom_right_video_path)

    fps = top_left_reader.get_meta_data()["fps"]  # or handle if they differ

    with imageio.get_writer(output_path, fps=fps) as writer:
        while True:
            try:
                tl_frame = top_left_reader.get_next_data()
                tr_frame = top_right_reader.get_next_data()
                bl_frame = bottom_left_reader.get_next_data()
                br_frame = bottom_right_reader.get_next_data()
            except (StopIteration, IndexError):
                # Means at least one reader had no more frames
                break

            # Combine frames in a quadrant layout
            top = np.hstack([tl_frame, tr_frame])
            bottom = np.hstack([bl_frame, br_frame])
            combined = np.vstack([top, bottom])

            writer.append_data(combined)

    top_left_reader.close()
    top_right_reader.close()
    bottom_left_reader.close()
    bottom_right_reader.close()


def quaternion_to_rotation_matrix(q):
    """Converts quaternion [w,x,y,z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),         2*(x*z + y*w)],
        [2*(x*y + z*w),           1 - 2*(x**2 + z**2),   2*(y*z - x*w)],
        [2*(x*z - y*w),           2*(y*z + x*w),         1 - 2*(x**2 + y**2)]
    ], dtype=np.float32)


def pose_vector_to_matrix(translation, quaternion):
    """Combines translation vector and quaternion into 4x4 transformation matrix."""
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = quaternion_to_rotation_matrix(quaternion)
    T[:3, 3]  = translation
    return T


def convert_mp4_to_webp(input_path, output_path, quality=80):
    """Converts MP4 video to WebP format using ffmpeg."""
    import subprocess
    import shutil
    
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg not found. Install via: sudo apt-get install ffmpeg")
    
    cmd = [
        ffmpeg_path, '-i', input_path,
        '-c:v', 'libwebp',
        '-quality', str(quality),
        '-lossless', '0',
        '-compression_level', '6',
        '-qmin', '0',
        '-qmax', '100',
        '-preset', 'default',
        '-loop', '0',
        '-vsync', '0',
        '-f', 'webp',
        output_path
    ]
    subprocess.run(cmd, check=True)
    

COLOR_RANGES = {
    "green":   (np.array([50, 150, 50],  dtype=np.uint8), np.array([70, 255, 255], dtype=np.uint8)),
    "cyan":    (np.array([80, 150, 50],  dtype=np.uint8), np.array([100,255,255],  dtype=np.uint8)),
    "magenta": (np.array([140, 150, 50], dtype=np.uint8), np.array([170,255,255],  dtype=np.uint8)),
}

def find_color_bounding_box(
    rgb_image: np.ndarray,
    color_name: str = "green",
    kernel_size: int = 3,
    erode_iters: int = 1,
    dilate_iters: int = 1
) -> tuple:
    """
    Finds the largest bounding box for a contiguous region of a specified color
    (by default, 'green') in an RGB image.

    Args:
        rgb_image: (H, W, 3) np.uint8 array in [0..255], representing an RGB image.
        color_name: 'green', 'cyan', or 'magenta' (or any color you have in COLOR_RANGES).
        kernel_size: size of morphological kernel for noise removal.
        erode_iters: how many times to erode.
        dilate_iters: how many times to dilate.

    Returns:
        bounding box as (x_min, y_min, x_max, y_max) around the largest
        contiguous blob of that color, or None if no blob is found.
    """
    # 1) Convert from RGB to HSV
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    # 2) Retrieve the lower & upper HSV bounds for the desired color.
    #    If color_name not in dict, raise error or handle it somehow
    if color_name not in COLOR_RANGES:
        raise ValueError(f"Unknown color '{color_name}'. Choose from {list(COLOR_RANGES.keys())}.")

    lower_hsv, upper_hsv = COLOR_RANGES[color_name]

    # 3) Create a mask for pixels within this HSV range
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # 4) Morphological operations to remove small noise
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=erode_iters)
    mask = cv2.dilate(mask, kernel, iterations=dilate_iters)

    # 5) Find connected components (largest color blob)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # stats shape: (num_labels, 5) => [label, left, top, width, height, area]
    # label=0 is background

    if num_labels <= 1:
        # No colored component found
        return None

    # 6) Identify the largest non-background component by area
    #    np.argmax(stats[1:, cv2.CC_STAT_AREA]) gives the largest blob among labels [1..]
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    # 7) Extract bounding box of that largest component
    x = stats[largest_label, cv2.CC_STAT_LEFT]
    y = stats[largest_label, cv2.CC_STAT_TOP]
    w = stats[largest_label, cv2.CC_STAT_WIDTH]
    h = stats[largest_label, cv2.CC_STAT_HEIGHT]

    # Return in (x_min, y_min, x_max, y_max)
    return (x, y, x + w, y + h)


def bounding_box_center(bbox):
    """
    Given a bbox = [x_min, y_min, x_max, y_max],
    return its center (cx, cy).
    """
    x_min, y_min, x_max, y_max = bbox
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    return cx, cy

def bounding_box_distance(bbox1, bbox2):
    """
    Compute the Euclidean distance between the centers of two bounding boxes.
    """
    cx1, cy1 = bounding_box_center(bbox1)
    cx2, cy2 = bounding_box_center(bbox2)
    return math.hypot(cx2 - cx1, cy2 - cy1)


def imitate_trajectory_with_action_identifier(
    dataset_path="/home/yilong/Documents/policy_data/lift/lift_smaller_2000",
    hand_mesh_dir="/home/yilong/Documents/action_extractor/action_extractor/megapose/panda_hand_mesh",
    output_dir="/home/yilong/Documents/action_extractor/debug/megapose_lift_smaller_2000",
    num_demos=100,
    save_webp=False,
    cameras=["frontview_image", "sideview_image"],
    batch_size=40,
):
    """
    - Only loads the Megapose model once (caching).
    - Suppresses Panda3D logging noise.
    - Uses the ActionIdentifierMegapose class to infer (n-1) actions from a sequence of frames,
      preserving exactly the same final logic (rolled-out environment, quadrant videos, etc.).
    """

    # ----------------------------------------------------------------
    # 0) Prepare output directory
    # ----------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # 1) Build object_dataset from mesh_dir
    # ----------------------------------------------------------------
    hand_object_dataset = make_object_dataset_from_folder(Path(hand_mesh_dir))

    # ----------------------------------------------------------------
    # 2) Load the model once
    # ----------------------------------------------------------------
    model_name = "megapose-1.0-RGB-multi-hypothesis"
    model_info = NAMED_MODELS[model_name]

    from megapose.utils.logging import get_logger
    logger = get_logger(__name__)
    logger.info(f"Loading model {model_name} once at script start.")

    hand_pose_estimator = load_named_model(model_name, hand_object_dataset).cuda()
    # In your old code, you also loaded finger_pose_estimator but you are only
    # using bounding boxes for fingers. That’s unchanged if you wish.

    # ----------------------------------------------------------------
    # 3) Preprocess dataset if needed (HDF5 -> Zarr)
    # ----------------------------------------------------------------
    sequence_dirs = glob(f"{dataset_path}/**/*.hdf5", recursive=True)
    for seq_dir in sequence_dirs:
        ds_dir = seq_dir.replace(".hdf5", ".zarr")
        zarr_path = seq_dir.replace(".hdf5", ".zarr.zip")
        if not os.path.exists(zarr_path):
            hdf5_to_zarr_parallel_with_progress(seq_dir, max_workers=16)
            store = DirectoryStore(ds_dir)
            root_z = zarr.group(store, overwrite=False)
            store.close()
            directorystore_to_zarr_zip(ds_dir, zarr_path)
            shutil.rmtree(ds_dir)

    # ----------------------------------------------------------------
    # 4) Collect the Zarr files
    # ----------------------------------------------------------------
    zarr_files = glob(f"{dataset_path}/**/*.zarr.zip", recursive=True)
    stores = [ZipStore(zarr_file, mode="r") for zarr_file in zarr_files]
    roots = [zarr.group(store) for store in stores]

    # ----------------------------------------------------------------
    # 5) Initialize environment
    # ----------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_meta = get_env_metadata_from_dataset(dataset_path=sequence_dirs[0])
    obs_modality_specs = {
        "obs": {
            "rgb": cameras,
            "depth": [f"{cam.split('_')[0]}_depth" for cam in cameras],
        }
    }
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)

    # Create environment for camera0
    env_camera0 = create_env_from_metadata(env_meta=env_meta, render_offscreen=True)

    # Grab an example image to figure out camera shape
    example_image = roots[0]["data"]["demo_0"]["obs"]["frontview_image"][0]
    camera_height, camera_width = example_image.shape[:2]

    # Intrinsics, extrinsics from your script
    K = get_camera_intrinsic_matrix(
        env_camera0.env.sim,
        camera_name="frontview",
        camera_height=camera_height,
        camera_width=camera_width,
    )
    R_4x4 = get_camera_extrinsic_matrix(env_camera0.env.sim, camera_name="frontview")

    # Wrap environment with video
    env_camera0 = VideoRecordingWrapper(
        env_camera0,
        video_recoder=VideoRecorder.create_h264(
            fps=20,
            codec="h264",
            input_pix_fmt="rgb24",
            crf=22,
            thread_type="FRAME",
            thread_count=1,
        ),
        steps_per_render=1,
        width=camera_width,
        height=camera_height,
        mode="rgb_array",
        camera_name=cameras[0].split("_")[0],
    )

    # Create environment for camera1
    env_camera1 = create_env_from_metadata(env_meta=env_meta, render_offscreen=True)
    env_camera1 = VideoRecordingWrapper(
        env_camera1,
        video_recoder=VideoRecorder.create_h264(
            fps=20,
            codec="h264",
            input_pix_fmt="rgb24",
            crf=22,
            thread_type="FRAME",
            thread_count=1,
        ),
        steps_per_render=1,
        width=camera_width,
        height=camera_height,
        mode="rgb_array",
        camera_name=cameras[1].split("_")[0],
    )

    # ----------------------------------------------------------------
    # 6) Build the ActionIdentifierMegapose
    # ----------------------------------------------------------------
    # This will replicate the chunk-based approach in a simpler method call.
    from action_extractor.megapose.action_identifier_megapose import ActionIdentifierMegapose

    action_identifier = ActionIdentifierMegapose(
        pose_estimator=hand_pose_estimator,
        R=R_4x4,  # from get_camera_extrinsic_matrix
        K=K,      # from get_camera_intrinsic_matrix
        model_info=model_info,
        batch_size=batch_size,   # chunk size
        scale_translation=80.0,  # same 80 factor
    )

    n_success = 0
    total_n = 0
    results = []

    # ----------------------------------------------------------------
    # 7) Main loop over demos
    # ----------------------------------------------------------------
    for root_z in roots:
        demos = list(root_z["data"].keys())[:num_demos] if num_demos else list(root_z["data"].keys())

        for demo in tqdm(demos, desc="Processing demos"):
            demo_id = demo.replace("demo_", "")
            upper_left_video_path  = os.path.join(output_dir, f"{demo_id}_upper_left.mp4")
            upper_right_video_path = os.path.join(output_dir, f"{demo_id}_upper_right.mp4")
            lower_left_video_path  = os.path.join(output_dir, f"{demo_id}_lower_left.mp4")
            lower_right_video_path = os.path.join(output_dir, f"{demo_id}_lower_right.mp4")
            combined_video_path    = os.path.join(output_dir, f"{demo_id}_combined.mp4")

            obs_group = root_z["data"][demo]["obs"]
            num_samples = obs_group["frontview_image"].shape[0]

            # ------------------------------
            # Build the "left" videos
            # ------------------------------
            upper_left_frames = [obs_group["frontview_image"][i] for i in range(num_samples)]
            lower_left_frames = [obs_group["sideview_image"][i]   for i in range(num_samples)]
            with imageio.get_writer(upper_left_video_path, fps=20) as writer:
                for frame in upper_left_frames:
                    writer.append_data(frame)
            with imageio.get_writer(lower_left_video_path, fps=20) as writer:
                for frame in lower_left_frames:
                    writer.append_data(frame)

            # ------------------------------
            # Use the new class to get poses + distances, then compute actions
            # ------------------------------
            frames_list = [obs_group["frontview_image"][i] for i in range(num_samples)]

            all_hand_poses_world, all_fingers_distances = action_identifier.get_all_hand_poses_finger_distances(frames_list)
            actions_for_demo = action_identifier.compute_actions(all_hand_poses_world, all_fingers_distances)

            # ------------------------------
            # Roll out environment with these actions
            # ------------------------------
            initial_state = root_z["data"][demo]["states"][0]

            # Right top video
            env_camera0.reset()
            env_camera0.reset_to({"states": initial_state})
            env_camera0.file_path = upper_right_video_path
            env_camera0.step_count = 0
            for act in actions_for_demo:
                env_camera0.step(act)
            env_camera0.video_recoder.stop()
            env_camera0.file_path = None

            # Right bottom video
            env_camera1.reset()
            env_camera1.reset_to({"states": initial_state})
            env_camera1.file_path = lower_right_video_path
            env_camera1.step_count = 0
            for act in actions_for_demo:
                env_camera1.step(act)
            env_camera1.video_recoder.stop()
            env_camera1.file_path = None

            # success check
            success = env_camera0.is_success()["task"] and env_camera1.is_success()["task"]
            if success:
                n_success += 1
            total_n += 1
            results.append(f"{demo}: {'success' if success else 'failed'}")

            # Combine quadrant videos
            combine_videos_quadrants(
                upper_left_video_path,
                upper_right_video_path,
                lower_left_video_path,
                lower_right_video_path,
                combined_video_path
            )
            os.remove(upper_left_video_path)
            os.remove(upper_right_video_path)
            os.remove(lower_left_video_path)
            os.remove(lower_right_video_path)

    # ----------------------------------------------------------------
    # 8) Final summary
    # ----------------------------------------------------------------
    success_rate = (n_success / total_n) * 100 if total_n else 0
    results.append(f"\nFinal Success Rate: {n_success}/{total_n}: {success_rate:.2f}%")

    with open(os.path.join(output_dir, "trajectory_results.txt"), "w") as f:
        f.write("\n".join(results))

    # ----------------------------------------------------------------
    # 9) Optionally convert MP4 => WEBP
    # ----------------------------------------------------------------
    if save_webp:
        print("Converting videos to webp format...")
        mp4_files = glob(os.path.join(output_dir, "*.mp4"))
        for mp4_file in tqdm(mp4_files, desc="Converting to webp"):
            webp_file = mp4_file.replace(".mp4", ".webp")
            try:
                convert_mp4_to_webp(mp4_file, webp_file)
                os.remove(mp4_file)
            except Exception as e:
                print(f"Error converting {mp4_file}: {e}")


if __name__ == "__main__":
    # Now you won't see the model reloaded every time, 
    # and Panda3D logs are suppressed to fatal.
    imitate_trajectory_with_action_identifier(
        dataset_path="/home/yilong/Documents/policy_data/lift/raw/1736991916_9054875/test",
        hand_mesh_dir="/home/yilong/Documents/action_extractor/action_extractor/megapose/panda_hand_mesh",
        output_dir="/home/yilong/Documents/action_extractor/debug/megapose_lift_smaller_2000",
        num_demos=100,
        save_webp=False,
        batch_size=40
    )
