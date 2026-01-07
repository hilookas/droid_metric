import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import json


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
    key,
    plot_in_world=False,
    load_optimized=False
):
    with open(key + "/_processor_solved_traj.json", "r") as fd:
        (
            Tactualcenter2cam_gripper_rights, interpolated_gripper_rights,
            Tactualcenter2cam_gripper_lefts, interpolated_gripper_lefts,
            Tactualcenter2cam_anchors, interpolated_anchors,
            Tactualcenter2cam_anchor_auxs, interpolated_anchor_auxs,
            Tanchor_aux2anchor,
        ) = json.load(fd)

    with open(key + "/_processor_optimized_traj.json", "r") as fd: # Overwrite
        (
            Tactualcenter2cam_gripper_rights_optimized,
            Tactualcenter2cam_gripper_lefts_optimized,
            Tactualcenter2cam_anchors_optimized,
            Tactualcenter2cam_anchor_auxs_optimized,
        ) = json.load(fd)

    slam_path = "/mnt/bn/ic-vlm/personal/cuihaiqin/droid_metric/output/poses"

    from glob import glob
    cam_pose_by_slam = []
    for path in sorted(glob(slam_path + "/*.txt")):
        cam_pose_by_slam.append(np.loadtxt(path))

    cam_pose_by_slam = np.array(cam_pose_by_slam)  # (N, 4, 4)

    # plot here
    # 3D trajectory plot for all transforms

    cam_pose_by_anchor = Tactualcenter2cam_anchors[0] @ np.linalg.inv(Tactualcenter2cam_anchors)
    cam_pose_by_anchor_aux = Tactualcenter2cam_anchor_auxs[0] @ np.linalg.inv(Tactualcenter2cam_anchor_auxs)
    cam_pose_by_anchor_optimized = Tactualcenter2cam_anchors_optimized[0] @ np.linalg.inv(Tactualcenter2cam_anchors_optimized)
    cam_pose_by_anchor_aux_optimized = Tactualcenter2cam_anchor_auxs_optimized[0] @ np.linalg.inv(Tactualcenter2cam_anchor_auxs_optimized)

    series = [
        ("anchor", cam_pose_by_anchor, (0.2, 0.2, 0.7)),
        ("anchor_aux", cam_pose_by_anchor_aux, (0.7, 0.7, 0.2)),
        ("anchor_optimized", cam_pose_by_anchor_optimized, (0.2, 0.7, 0.7)),
        ("anchor_aux_optimized", cam_pose_by_anchor_aux_optimized, (0.7, 0.2, 0.2)),
        ("slam", cam_pose_by_slam, (0.2, 0.7, 0.2)),
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
    plt.savefig('debug_t2cam_trajectories.png', dpi=150)

    # 2D plots for rotation components (roll, pitch, yaw)
    fig_rot, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    component_labels = ["roll (rad)", "pitch (rad)", "yaw (rad)"]

    for name, transforms, base_color in series:
        rolls, pitches, yaws = extract_rpy_sequences(transforms)
        for idx, comp in enumerate([rolls, pitches, yaws]):
            axs[idx].plot(comp, color=base_color, label=name)

    for idx, ax_comp in enumerate(axs):
        ax_comp.set_ylabel(component_labels[idx])
        ax_comp.legend()
        ax_comp.grid(True, linestyle='--', alpha=0.3)

    axs[-1].set_xlabel('Frame')
    fig_rot.tight_layout()
    fig_rot.savefig('debug_rotations.png', dpi=150)

    # 2D plots for translation components (x, y, z)
    fig_pos, axs_pos = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    pos_labels = ["x (m)", "y (m)", "z (m)"]

    for name, transforms, base_color in series:
        xs, ys, zs = extract_xyz_sequences(transforms)
        for idx, comp in enumerate([xs, ys, zs]):
            axs_pos[idx].plot(comp, color=base_color, label=name)

    for idx, ax_pos in enumerate(axs_pos):
        ax_pos.set_ylabel(pos_labels[idx])
        ax_pos.legend()
        ax_pos.grid(True, linestyle='--', alpha=0.3)

    axs_pos[-1].set_xlabel('Frame')
    fig_pos.tight_layout()
    fig_pos.savefig('debug_positions.png', dpi=150)

    print('Saved 3D trajectory plot to debug_t2cam_trajectories.png')
    print('Saved rotation component plots to debug_rotations.png')
    print('Saved translation component plots to debug_positions.png')
    plt.show()

    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "key",
        type=str,
    )
    parser.add_argument(
        "--in-world",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--optimized",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()
    plot_traj(args.key, plot_in_world=args.in_world, load_optimized=args.optimized)
