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

# video (unlabeled) -> use pose estimation to label videos -> video with pseudo labels -> train policy

# This script:
# 1) Takes video from dataset
# 2) Pass video into megapose to get pose estimations
# 3) Use pose estimations to get actions
# 4) Use actions to roll-out in the simulator
# 5) Save video into designated mp4 file
# Function: evaluate quality of pose estimation

#####################################################################
# Normal imports
#####################################################################
import os
import numpy as np
import torch
import math
import imageio
import cv2
import zarr
import shutil
from glob import glob
from tqdm import tqdm
from pathlib import Path
from zarr import DirectoryStore, ZipStore
import itertools

import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.file_utils import get_env_metadata_from_dataset
from robomimic.utils.env_utils import create_env_from_metadata

from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

# Megapose
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.utils.logging import get_logger
logger = get_logger(__name__)

# Our local imports
from action_extractor.utils.dataset_utils import (
    hdf5_to_zarr_parallel_with_progress,
    directorystore_to_zarr_zip,
)

from action_extractor.utils.angles import *

from action_extractor.megapose.action_identifier_megapose import (
    make_object_dataset_from_folder,
    bounding_box_center,   # might be unused
    find_color_bounding_box,
    pixel_to_world,
    poses_to_absolute_actions,
    poses_to_absolute_actions_average,
    poses_to_absolute_actions_mixed_ori_v1,
    ActionIdentifierMegapose
)
from robosuite.utils.camera_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
)

from transforms3d.euler import quat2euler, euler2quat

from scipy.spatial.transform import Rotation as R


def combine_videos_quadrants(top_left_video_path, top_right_video_path, 
                             bottom_left_video_path, bottom_right_video_path, 
                             output_path):
    """
    Combines four videos into a single quadrant layout video. 
    Continues until the *longest* video ends.
    If a shorter video ends, we freeze its last frame until all videos are done.
    """
    readers = [
        imageio.get_reader(top_left_video_path),
        imageio.get_reader(top_right_video_path),
        imageio.get_reader(bottom_left_video_path),
        imageio.get_reader(bottom_right_video_path)
    ]

    # We'll assume the FPS of the first video, but you can adapt if they differ.
    fps = readers[0].get_meta_data().get("fps", 20)

    # Keep track of whether each video is done reading
    done = [False, False, False, False]
    # Store the last frame for each quadrant
    last_frames = [None, None, None, None]

    # Initialize each video with its first frame if possible
    for i in range(4):
        try:
            last_frames[i] = readers[i].get_next_data()
        except (StopIteration, IndexError):
            # If no frame, mark done and last_frames[i] = None
            done[i] = True
            last_frames[i] = None

    with imageio.get_writer(output_path, fps=fps) as writer:
        while True:
            # If all are done, stop
            if all(done):
                break

            # Attempt to read the next frame from each video not yet done
            for i in range(4):
                if not done[i]:
                    try:
                        new_frame = readers[i].get_next_data()
                        last_frames[i] = new_frame
                    except (StopIteration, IndexError):
                        # Mark that video as done; keep last frame frozen
                        done[i] = True

            # At this point, we have an updated 'last_frames' for each quadrant
            # Some might be frozen if that video is done

            # If any of the last_frames is None from the start, create a black image 
            # matching the shape of a non-None frame. If *all* are None, we have no data left.
            if all(frame is None for frame in last_frames):
                # Means all videos had 0 frames from the start, or we used them up
                break

            # For a None frame (no data ever), freeze as black image matching shape of any valid frame
            # We'll find the first valid shape
            shape_for_black = None
            for f in last_frames:
                if f is not None:
                    shape_for_black = f.shape
                    break
            if shape_for_black is None:
                # No valid shape at all => end
                break

            # For any quadrant that is None, produce a black image of the same shape
            for i in range(4):
                if last_frames[i] is None:
                    last_frames[i] = np.zeros(shape_for_black, dtype=np.uint8)

            tl_frame = last_frames[0]
            tr_frame = last_frames[1]
            bl_frame = last_frames[2]
            br_frame = last_frames[3]

            # Combine frames in a quadrant layout
            top = np.hstack([tl_frame, tr_frame])
            bottom = np.hstack([bl_frame, br_frame])
            combined = np.vstack([top, bottom])

            writer.append_data(combined)

    # Close all readers
    for r in readers:
        r.close()


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

def load_ground_truth_poses_as_actions(obs_group, env_camera0):
    """
    Modified version:
      - We compute delta quaternions between consecutive steps, 
      - then 'add' (i.e. multiply) that delta with 'current_orientation'.
      - Then convert that updated orientation to axis-angle for each action.
    """

    # (N, 3)
    pos_array = obs_group["robot0_eef_pos"][:]   # end-effector positions
    # (N, 4) => [qx, qy, qz, qw] in the world frame
    quat_array = obs_group["robot0_eef_quat"][:] 

    num_samples = pos_array.shape[0]

    # Some internal environment variable -- adjust if needed.
    current_orientation = env_camera0.env.env._eef_xquat.astype(np.float32)
    current_orientation = quat_normalize(current_orientation)

    # We will have one fewer action than the number of samples,
    # because we form deltas between consecutive frames.
    num_actions = num_samples - 1

    all_actions = np.zeros((num_actions, 7), dtype=np.float32)

    # Keep track of the previous axis-angle to unify sign across consecutive frames
    prev_rvec = None

    for i in range(num_actions):
        q_i   = quat_array[i]
        q_i1  = quat_array[i + 1]
        q_i   = quat_normalize(q_i)
        q_i1  = quat_normalize(q_i1)

        # 1) compute delta: how do we go from q_i to q_i1?
        #    delta = q_i1 * inv(q_i)
        q_delta = quat_multiply(quat_inv(q_i), q_i1)
        q_delta = quat_normalize(q_delta)

        # 2) "add" this delta to current_orientation (i.e. multiply in quaternion space)
        current_orientation = quat_multiply(current_orientation, q_delta)
        current_orientation = quat_normalize(current_orientation)

        # 3) Convert to axis-angle
        rvec = quat2axisangle(current_orientation)

        # 4) optional sign unify (avoid ± flips in axis representation)
        if prev_rvec is not None and np.dot(rvec, prev_rvec) < 0:
            rvec = -rvec
        prev_rvec = rvec

        # For position, you can choose either pos_array[i] or pos_array[i+1]. 
        # Typically we take the next position: 
        px, py, pz = pos_array[i+1]

        all_actions[i, 0:3] = [px, py, pz]
        all_actions[i, 3:6] = rvec
        all_actions[i, 6]   = 1.0  # e.g., keep gripper = open = 1

    return all_actions

def imitate_trajectory_with_action_identifier(
    dataset_path="/home/yilong/Documents/policy_data/lift/lift_smaller_2000",
    hand_mesh_dir="/home/yilong/Documents/action_extractor/action_extractor/megapose/panda_hand_mesh",
    output_dir="/home/yilong/Documents/action_extractor/debug/megapose_lift_smaller_2000",
    num_demos=100,
    save_webp=False,
    cameras=["squared0view_image", "squared0view2_image"],  # now general "camA_image" & "camB_image"
    batch_size=40,
):
    """
    General version where 'cameras' is any 2-element list of camera angle strings, 
    e.g. ["frontview_image", "sideview_image"] or ["birdview_image","agentview_image"].
    This code references cameras[0] -> camera A, cameras[1] -> camera B.
    """

    # 0) Output dir
    os.makedirs(output_dir, exist_ok=True)

    # 1) Build object dataset
    hand_object_dataset = make_object_dataset_from_folder(Path(hand_mesh_dir))

    # 2) Load model once
    model_name = "megapose-1.0-RGB-multi-hypothesis"
    model_info = NAMED_MODELS[model_name]
    logger.info(f"Loading model {model_name} once at script start.")
    hand_pose_estimator = load_named_model(model_name, hand_object_dataset).cuda()

    # 3) Preprocess dataset => zarr
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

    # 4) Collect Zarr files
    zarr_files = glob(f"{dataset_path}/**/*.zarr.zip", recursive=True)
    stores = [ZipStore(zarr_file, mode="r") for zarr_file in zarr_files]
    roots = [zarr.group(store) for store in stores]

    # 5) Create environment
    env_meta = get_env_metadata_from_dataset(dataset_path=sequence_dirs[0])
    env_meta['env_kwargs']['controller_configs']['control_delta'] = False
    env_meta['env_kwargs']['controller_configs']['type'] = 'OSC_POSE'

    # We'll define "camera0" => cameras[0], "camera1" => cameras[1]
    camera0_name = cameras[0].split("_")[0]
    camera1_name = cameras[1].split("_")[0]

    obs_modality_specs = {
        "obs": {
            "rgb": cameras,  # e.g. ["frontview_image","sideview_image"]
            "depth": [f"{cam.split('_')[0]}_depth" for cam in cameras],
        }
    }
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)

    env_camera0 = create_env_from_metadata(env_meta=env_meta, render_offscreen=True)

    # Example image from camera0
    example_image = roots[0]["data"]["demo_0"]["obs"][cameras[0]][0]
    camera_height, camera_width = example_image.shape[:2]

    # We fetch intrinsics & extrinsics for these two cameras:
    camera0_K = get_camera_intrinsic_matrix(env_camera0.env.sim,
                                            camera_name=camera0_name,
                                            camera_height=camera_height,
                                            camera_width=camera_width)
    camera1_K = get_camera_intrinsic_matrix(env_camera0.env.sim,
                                            camera_name=camera1_name,
                                            camera_height=camera_height,
                                            camera_width=camera_width)
    camera0_R = get_camera_extrinsic_matrix(env_camera0.env.sim, camera_name=camera0_name)
    camera1_R = get_camera_extrinsic_matrix(env_camera0.env.sim, camera_name=camera1_name)

    env_camera0 = VideoRecordingWrapper(
        env_camera0,
        video_recoder=VideoRecorder.create_h264(fps=20, codec="h264", input_pix_fmt="rgb24", crf=22),
        steps_per_render=1,
        width=camera_width,
        height=camera_height,
        mode="rgb_array",
        camera_name=camera0_name,
    )

    # We'll do a second environment for camera1
    env_camera1 = create_env_from_metadata(env_meta=env_meta, render_offscreen=True)
    env_camera1 = VideoRecordingWrapper(
        env_camera1,
        video_recoder=VideoRecorder.create_h264(fps=20, codec="h264", input_pix_fmt="rgb24", crf=22),
        steps_per_render=1,
        width=camera_width,
        height=camera_height,
        mode="rgb_array",
        camera_name=camera1_name,
    )

    # 6) Build the ActionIdentifierMegapose using camera0/camera1 info
    action_identifier = ActionIdentifierMegapose(
        pose_estimator=hand_pose_estimator,
        cameraA_R=camera0_R,
        cameraA_K=camera0_K,
        cameraB_R=camera1_R,
        cameraB_K=camera1_K,
        model_info=model_info,
        batch_size=batch_size,
        scale_translation=80.0,
    )

    n_success = 0
    total_n = 0
    results = []

    # 7) Loop over demos
    for root_z in roots:
        demos = list(root_z["data"].keys())[:num_demos] if num_demos else list(root_z["data"].keys())
        for demo in tqdm(demos, desc="Processing demos"):
            demo_id = demo.replace("demo_", "")
            upper_left_video_path  = os.path.join(output_dir, f"{demo_id}_upper_left.mp4")
            upper_right_video_path = os.path.join(output_dir, f"{demo_id}_upper_right.mp4")
            lower_left_video_path  = os.path.join(output_dir, f"{demo_id}_lower_left.mp4")
            lower_right_video_path = os.path.join(output_dir, f"{demo_id}_lower_right.mp4")
            combined_video_path    = os.path.join(output_dir, f"{demo_id}_combined.mp4")

            obs_group   = root_z["data"][demo]["obs"]
            num_samples = obs_group[cameras[0]].shape[0]

            # Let's define camera0_frames, camera1_frames
            camera0_frames = [obs_group[cameras[0]][i] for i in range(num_samples)]
            camera1_frames = [obs_group[cameras[1]][i] for i in range(num_samples)]

            # We'll produce "upper_left" from camera0, "lower_left" from camera1
            with imageio.get_writer(upper_left_video_path, fps=20) as writer:
                for frame in camera0_frames:
                    writer.append_data(frame)
            with imageio.get_writer(lower_left_video_path, fps=20) as writer:
                for frame in camera1_frames:
                    writer.append_data(frame)

            # Depth lists if present
            camera0_depth_list = [obs_group[f"{camera0_name}_depth"][i] for i in range(num_samples)]
            camera1_depth_list = [obs_group[f"{camera1_name}_depth"][i] for i in range(num_samples)]

            cache_file = f"hand_poses_{camera0_name}_{camera1_name}_cache.npz"
            if os.path.exists(cache_file):
                # Load from disk
                print(f"Loading cached poses from {cache_file} ...")
                data = np.load(cache_file, allow_pickle=True)
                all_hand_poses_camA = data["all_hand_poses_camA"]
                all_hand_poses_camB = data["all_hand_poses_camB"]
            else:
                # No cache => run the expensive inference once
                print(f"No cache found. Running inference to get poses for camera {camera0_name}/{camera1_name} ...")
                # We'll call the new get_poses_from_frames method
                (all_hand_poses_camA,
                 all_hand_poses_camB) = action_identifier.get_poses_from_frames(
                     cameraA_frames_list=camera0_frames,
                     cameraA_depth_list=camera0_depth_list,
                     cameraB_frames_list=camera1_frames,  # optional
                 )
                # Save to disk for future runs
                np.savez(
                    cache_file,
                    all_hand_poses_camA=all_hand_poses_camA,
                    all_hand_poses_camB=all_hand_poses_camB
                )
            
            gt_gripper_actions = [root_z["data"][demo]['actions'][i][-1] for i in range(num_samples)]
           
            # Build your absolute actions from all_hand_poses_camA, all_hand_poses_camB
            actions_for_demo = poses_to_absolute_actions_average(
                all_hand_poses_camA,
                all_hand_poses_camB,
                gt_gripper_actions,
                env_camera0
            )

            initial_state = root_z["data"][demo]["states"][0]

            # top-right video => camera0 environment
            env_camera0.reset()
            env_camera0.reset_to({"states": initial_state})
            env_camera0.file_path = upper_right_video_path
            env_camera0.step_count = 0
            for action in actions_for_demo:
                env_camera0.step(action)
            env_camera0.video_recoder.stop()
            env_camera0.file_path = None

            # bottom-right video => camera1 environment
            env_camera1.reset()
            env_camera1.reset_to({"states": initial_state})
            env_camera1.file_path = lower_right_video_path
            env_camera1.step_count = 0
            for action in actions_for_demo:
                env_camera1.step(action)
            env_camera1.video_recoder.stop()
            env_camera1.file_path = None

            # success check
            success = env_camera0.is_success()["task"]
            if success:
                n_success += 1
            total_n += 1
            results.append(f"{demo}: {'success' if success else 'failed'}")

            # Combine quadrant
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

    success_rate = (n_success / total_n)*100 if total_n else 0
    results.append(f"\nFinal Success Rate: {n_success}/{total_n} => {success_rate:.2f}%")

    with open(os.path.join(output_dir, "trajectory_results.txt"), "w") as f:
        f.write("\n".join(results))

    if save_webp:
        print("Converting to webp...")
        mp4_files = glob(os.path.join(output_dir, "*.mp4"))
        for mp4_file in tqdm(mp4_files):
            webp_file = mp4_file.replace(".mp4", ".webp")
            try:
                convert_mp4_to_webp(mp4_file, webp_file)
                os.remove(mp4_file)
            except Exception as e:
                print(f"Error converting {mp4_file}: {e}")

    print(f"Wrote results to {os.path.join(output_dir, 'trajectory_results.txt')}")


if __name__ == "__main__":
    imitate_trajectory_with_action_identifier(
        dataset_path="/home/yilong/Documents/policy_data/square_d0/raw/test/test_tableview",
        hand_mesh_dir="/home/yilong/Documents/action_extractor/action_extractor/megapose/panda_hand_mesh",
        output_dir="/home/yilong/Documents/action_extractor/debug/megapose_average_frontsideviews",
        num_demos=3,
        save_webp=False,
        batch_size=40
    )