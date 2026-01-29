import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import json
from pathlib import Path
from datetime import datetime

TARGET_LOCATION = "/mnt/hdfs/haruna/home/byte_data_seed/hdd_hldy/vla/users/cuihaiqin/tvla-data"


def rotation_matrix_to_rpy(R):
    """Convert a 3x3 rotation matrix to roll-pitch-yaw angles (XYZ order)."""
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def extract_rpy_sequences(transforms):
    rolls, pitches, yaws = [], [], []
    for T in transforms:
        r, p, y = rotation_matrix_to_rpy(T[:3, :3])
        rolls.append(r)
        pitches.append(p)
        yaws.append(y)
    return np.array(rolls), np.array(pitches), np.array(yaws)


def extract_xyz_sequences(transforms):
    xs, ys, zs = [], [], []
    for T in transforms:
        xs.append(T[0, 3])
        ys.append(T[1, 3])
        zs.append(T[2, 3])
    return np.array(xs), np.array(ys), np.array(zs)

def plot_traj(
    key
):
    with open(Path(TARGET_LOCATION) / key / "_processor_optimized_traj.json", "r") as fd:
        [
            Tactualcenter2cam_gripper_right_preds,
            Tactualcenter2cam_gripper_left_preds,
            Tactualcenter2cam_anchor_preds,
            Tactualcenter2cam_anchor_aux_preds,
        ] = json.load(fd)

    # from pytransform3d.rotations import axis_angle_from_matrix
    # Tgripperright2world = (np.linalg.inv(np.array(Tactualcenter2cam_gripper_right_preds[0])) @ np.array(Tactualcenter2cam_anchor_preds[0]))
    # import math
    # print(math.atan(-Tgripperright2world[2,0]/-Tgripperright2world[0,0]))
    # import ipdb; ipdb.set_trace()
    # exit()

    with open(Path(TARGET_LOCATION) / key / "_processor_optimized_traj_slam.json", "r") as fd:
        [
            Tactualcenter2cam_gripper_right_preds_slam,
            Tactualcenter2cam_gripper_left_preds_slam,
        ] = json.load(fd)

    with open(Path(TARGET_LOCATION) / key / "_processor_cam_slam_traj.json", "r") as fd:
        cam_pose_by_slam = json.load(fd)

    Tgripperright2cams = np.array(Tactualcenter2cam_gripper_right_preds_slam)
    Tgripperleft2cams = np.array(Tactualcenter2cam_gripper_left_preds_slam)
    Tcam2firstcams = np.array(cam_pose_by_slam)

    with open(Path(TARGET_LOCATION) / key / "data.v1.json", "r") as f:
        data_old = json.load(f)

    import copy
    data = copy.deepcopy(data_old)

    data["arms"]["main"]["Ttcp2firsttcps"] = [(np.linalg.inv(Tcam2firstcams[0] @ Tgripperright2cams[0]) @ Tcam2firstcam @ np.array(Ttcp2cam)).tolist() for Ttcp2cam, Tcam2firstcam in zip(Tgripperright2cams, Tcam2firstcams)]
    data["arms"]["main"]["Tfirsttcp2maincam"] = Tgripperright2cams[0].tolist()
    data["arms"]["left"]["Ttcp2firsttcps"] = [(np.linalg.inv(Tcam2firstcams[0] @ Tgripperleft2cams[0]) @ Tcam2firstcam @ np.array(Ttcp2cam)).tolist() for Ttcp2cam, Tcam2firstcam in zip(Tgripperleft2cams, Tcam2firstcams)]
    data["arms"]["left"]["Tfirsttcp2maincam"] = Tgripperleft2cams[0].tolist()

    with open(Path(TARGET_LOCATION) / key / "data.v1_slam.json", "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # plot here
    # 3D trajectory plot for all transforms

    series = [
        ("anchor", np.array(data_old["arms"]["main"]["Ttcp2firsttcps"]), (0.2, 0.2, 0.7)),
        ("anchor_left", np.array(data_old["arms"]["left"]["Ttcp2firsttcps"]), (0.7, 0.7, 0.2)),
        ("slam", np.array(data["arms"]["main"]["Ttcp2firsttcps"]), (0.2, 0.7, 0.2)),
        ("slam_left", np.array(data["arms"]["left"]["Ttcp2firsttcps"]), (0.7, 0.7, 0.7)),
    ]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    for name, transforms, base_color in series:
        # extract_positions
        xs, ys, zs = [], [], []
        for T in transforms:
            xs.append(T[0][3]); ys.append(T[1][3]); zs.append(T[2][3])
        xs, ys, zs =  np.array(xs), np.array(ys), np.array(zs)

        # Optionally connect with lines for continuity
        ax.plot(xs, ys, zs, color=base_color, label=name)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(Path(TARGET_LOCATION) / key / "_processor_slam_hand_trajectories.png", dpi=150)

    # Combined 3x2 plots: left column XYZ, right column RPY
    fig_comb, axs_grid = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    pos_labels = ["x (m)", "y (m)", "z (m)"]
    rpy_labels = ["roll (rad)", "pitch (rad)", "yaw (rad)"]

    for name, transforms, base_color in series:
        xs, ys, zs = extract_xyz_sequences(transforms)
        rolls, pitches, yaws = extract_rpy_sequences(transforms)

        # Left column: XYZ
        axs_grid[0, 0].plot(xs, color=base_color, label=name)
        axs_grid[1, 0].plot(ys, color=base_color, label=name)
        axs_grid[2, 0].plot(zs, color=base_color, label=name)

        # Right column: RPY
        axs_grid[0, 1].plot(rolls, color=base_color, label=name)
        axs_grid[1, 1].plot(pitches, color=base_color, label=name)
        axs_grid[2, 1].plot(yaws, color=base_color, label=name)

    # Labels, legends, grids
    for i in range(3):
        axs_grid[i, 0].set_ylabel(pos_labels[i])
        axs_grid[i, 1].set_ylabel(rpy_labels[i])
        axs_grid[i, 0].legend()
        axs_grid[i, 1].legend()
        axs_grid[i, 0].grid(True, linestyle='--', alpha=0.3)
        axs_grid[i, 1].grid(True, linestyle='--', alpha=0.3)

    # Column titles
    axs_grid[0, 0].set_title('Position')
    axs_grid[0, 1].set_title('Rotation')

    # X labels for bottom row
    axs_grid[2, 0].set_xlabel('Frame')
    axs_grid[2, 1].set_xlabel('Frame')

    fig_comb.tight_layout()
    plt.savefig(Path(TARGET_LOCATION) / key / "_processor_slam_hand_rpy_xyz_combined.png", dpi=150)
    # plt.show()

    # import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "key",
        type=str,
    )
    args = parser.parse_args()
    plot_traj(args.key)
