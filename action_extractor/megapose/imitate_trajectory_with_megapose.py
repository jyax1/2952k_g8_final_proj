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
from action_extractor.megapose.action_identifier_megapose import (
    make_object_dataset_from_folder,
    bounding_box_center,   # might be unused
    find_color_bounding_box,
    pixel_to_world,
    ActionIdentifierMegapose
)
from robosuite.utils.camera_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
)

from scipy.spatial.transform import Rotation as R


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


def axisangle2quat(axis_angle):
    """
    Converts an axis-angle vector [rx, ry, rz] into a quaternion [x, y, z, w].
    """
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-12:
        # nearly zero rotation
        return np.array([0, 0, 0, 1], dtype=np.float32)
    axis = axis_angle / angle
    half = angle * 0.5
    return np.concatenate([
        axis * np.sin(half),
        [np.cos(half)]
    ]).astype(np.float32)


def quat2axisangle(quat):
    """
    Convert quaternion [x, y, z, w] to axis-angle [rx, ry, rz].
    """
    w = quat[3]
    # clamp w
    if w > 1.0:
        w = 1.0
    elif w < -1.0:
        w = -1.0

    angle = 2.0 * math.acos(w)
    den = math.sqrt(1.0 - w * w)
    if den < 1e-12:
        return np.zeros(3, dtype=np.float32)

    axis = quat[:3] / den
    return axis * angle


def quat_multiply(q1, q0):
    """
    Multiply two quaternions q1 * q0 in xyzw form => xyzw
    """
    x0, y0, z0, w0 = q0
    x1, y1, z1, w1 = q1
    return np.array([
        x1*w0 + y1*z0 - z1*y0 + w1*x0,
        -x1*z0 + y1*w0 + z1*x0 + w1*y0,
        x1*y0 - y1*x0 + z1*w0 + w1*z0,
        -x1*x0 - y1*y0 - z1*z0 + w1*w0,
    ], dtype=np.float32)


def quat_inv(q):
    """
    Inverse of unit quaternion [x, y, z, w] is the conjugate => [-x, -y, -z, w].
    """
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)


def load_ground_truth_poses_as_actions(obs_group, env_camera0):
    """
    Convert the dataset's end-effector poses (in world frame) into the orientation
    that OSC_POSE with absolute control actually wants, i.e. the "eef_site" frame orientation.

    1) We read robot0_eef_pos, robot0_eef_quat  => these are presumably world-frame
       orientation of the end effector.

    2) The environment's absolute action expects orientation in the "eef_site" coordinate system.

       Let q_offset = env_camera0.env.env.robots[0].eef_rot_offset
         (the rotation offset between link7 and eef site)
       Possibly also link7 orientation if needed, but typically we can do:
         q_site = q_world * inv(q_offset)     (assuming q_offset transforms link7->eef_site)

       Adjust as needed if your dataset is truly "world->gripper" vs "world->eef_site."

    3) Convert to axis-angle and unify sign if you want (optional).

    4) Return (N, 7) actions => [px, py, pz, rx, ry, rz, 1].
    """
    pos_array = obs_group["robot0_eef_pos"][:]    # shape (N,3)
    quat_array = obs_group["robot0_eef_quat"][:]  # shape (N,4) => [qx, qy, qz, qw] (world)
    num_samples = pos_array.shape[0]

    # eef_rot_offset is typically [x, y, z, w].  We want to transform the dataset's world orientation
    # into the "eef_site frame" orientation that the controller needs.
    q_offset = env_camera0.env.env._eef_xquat  # shape (4,)
    q_base = env_camera0.env.env.robots[0]._hand_quat

    all_actions = np.zeros((num_samples, 7), dtype=np.float32)

    # optional: unify sign across consecutive frames
    prev_rvec = None

    for i in range(num_samples):
        px, py, pz = pos_array[i]

        q_world = quat_array[i]  # [w, x, y, z]
        q_world = q_world[[1, 2, 3, 0]] # [x, y, z, w]
        q_eef_site = q_offset
        # convert to axis-angle
        rvec = quat2axisangle(q_eef_site)

        # unify sign across consecutive frames to avoid ± axis flips
        if (prev_rvec is not None) and (np.dot(rvec, prev_rvec) < 0):
            rvec = -rvec
        prev_rvec = rvec

        all_actions[i, :3] = [px, py, pz]
        all_actions[i, 3:6] = rvec
        all_actions[i, 6] = 1.0  # keep gripper=1 as "open," or do your logic

    return all_actions


def imitate_trajectory_with_action_identifier(
    dataset_path="/home/yilong/Documents/policy_data/lift/lift_smaller_2000",
    hand_mesh_dir="/home/yilong/Documents/action_extractor/action_extractor/megapose/panda_hand_mesh",
    output_dir="/home/yilong/Documents/action_extractor/debug/megapose_lift_smaller_2000",
    num_demos=100,
    save_webp=False,
    cameras=["frontview_image", "sideview_image"],
    batch_size=40,
):
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

    obs_modality_specs = {
        "obs": {
            "rgb": cameras,
            "depth": [f"{cam.split('_')[0]}_depth" for cam in cameras],
        }
    }
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)

    env_camera0 = create_env_from_metadata(env_meta=env_meta, render_offscreen=True)

    example_image = roots[0]["data"]["demo_0"]["obs"]["frontview_image"][0]
    camera_height, camera_width = example_image.shape[:2]

    frontview_K = get_camera_intrinsic_matrix(env_camera0.env.sim,
                                              camera_name="frontview",
                                              camera_height=camera_height,
                                              camera_width=camera_width)
    sideview_K   = get_camera_intrinsic_matrix(env_camera0.env.sim,
                                               camera_name="sideview",
                                               camera_height=camera_height,
                                               camera_width=camera_width)
    frontview_R  = get_camera_extrinsic_matrix(env_camera0.env.sim, camera_name="frontview")
    sideview_R   = get_camera_extrinsic_matrix(env_camera0.env.sim, camera_name="sideview")

    env_camera0 = VideoRecordingWrapper(
        env_camera0,
        video_recoder=VideoRecorder.create_h264(fps=20, codec="h264", input_pix_fmt="rgb24", crf=22),
        steps_per_render=1,
        width=camera_width,
        height=camera_height,
        mode="rgb_array",
        camera_name=cameras[0].split("_")[0],
    )

    env_camera1 = create_env_from_metadata(env_meta=env_meta, render_offscreen=True)
    env_camera1 = VideoRecordingWrapper(
        env_camera1,
        video_recoder=VideoRecorder.create_h264(fps=20, codec="h264", input_pix_fmt="rgb24", crf=22),
        steps_per_render=1,
        width=camera_width,
        height=camera_height,
        mode="rgb_array",
        camera_name=cameras[1].split("_")[0],
    )

    # 6) Optionally build the ActionIdentifierMegapose
    action_identifier = ActionIdentifierMegapose(
        pose_estimator=hand_pose_estimator,
        frontview_R=frontview_R,
        frontview_K=frontview_K,
        sideview_R=sideview_R,
        sideview_K=sideview_K,
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
            num_samples = obs_group["frontview_image"].shape[0]

            # Left videos
            upper_left_frames = [obs_group["frontview_image"][i] for i in range(num_samples)]
            lower_left_frames = [obs_group["sideview_image"][i] for i in range(num_samples)]
            with imageio.get_writer(upper_left_video_path, fps=20) as writer:
                for frame in upper_left_frames:
                    writer.append_data(frame)
            with imageio.get_writer(lower_left_video_path, fps=20) as writer:
                for frame in lower_left_frames:
                    writer.append_data(frame)

            # ---- We want ground-truth absolute actions in eef_site coords ----
            actions_for_demo = load_ground_truth_poses_as_actions(obs_group, env_camera0)
            
            front_frames_list = [obs_group["frontview_image"][i] for i in range(num_samples)]
            front_depth_list = [obs_group["frontview_depth"][i] for i in range(num_samples)]
            side_frames_list = [obs_group["sideview_image"][i] for i in range(num_samples)]
            
            cache_file = "hand_poses_cache.npz"
            if os.path.exists(cache_file):
                # Load from disk
                print(f"Loading cached poses from {cache_file} ...")
                data = np.load(cache_file, allow_pickle=True)
                all_hand_poses_world = data["all_hand_poses_world"]
                all_fingers_distances = data["all_fingers_distances"]
                all_hand_poses_world_from_side = data["all_hand_poses_world_from_side"]
            else:
                # No cache => run the expensive inference once
                print("No cache found. Running inference to get all_hand_poses_world...")
                (all_hand_poses_world,
                all_fingers_distances,
                all_hand_poses_world_from_side) = action_identifier.get_all_hand_poses_finger_distances_with_side(
                    front_frames_list,
                    front_depth_list=None,
                    side_frames_list=side_frames_list
                )
                # Save to disk for future runs
                np.savez(
                    cache_file,
                    all_hand_poses_world=all_hand_poses_world,
                    all_fingers_distances=all_fingers_distances,
                    all_hand_poses_world_from_side=all_hand_poses_world_from_side
                )

           
            actions_for_demo = all_hand_poses_world
            

            initial_state = root_z["data"][demo]["states"][0]

            # 1) top-right video
            env_camera0.reset()
            env_camera0.reset_to({"states": initial_state})
            env_camera0.file_path = upper_right_video_path
            env_camera0.step_count = 0
            for action in actions_for_demo:
                env_camera0.step(action)
            env_camera0.video_recoder.stop()
            env_camera0.file_path = None

            # 2) bottom-right video
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

            # combine quadrant
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
        dataset_path="/home/yilong/Documents/policy_data/square_d0/raw/test/test",
        hand_mesh_dir="/home/yilong/Documents/action_extractor/action_extractor/megapose/panda_hand_mesh",
        output_dir="/home/yilong/Documents/action_extractor/debug/megapose_gt",
        num_demos=3,
        save_webp=False,
        batch_size=40
    )
