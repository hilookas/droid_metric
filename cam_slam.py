import json
import shutil
import subprocess
import uuid
from pathlib import Path

import numpy as np

TARGET_LOCATION = "/mnt/hdfs/haruna/home/byte_data_seed/hdd_hldy/vla/users/cuihaiqin/tvla-data"


def cam_slam(key: str) -> None:
    tmp_dir = Path(f"/tmp/cam_slam-{uuid.uuid4()}")
    print(str(tmp_dir))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(Path(TARGET_LOCATION) / Path(key) / "_processor_cam_param.json", "r") as fp:
            (
                width, height,
                intrinsic,
                distortion_coefficients
            ) = json.load(fp)
        np.savetxt(tmp_dir / "intrinsic.txt", np.array([intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2]], dtype=float))

        subprocess.run([
            "python",
            "reconstruct.py",
            "--input",
            str(Path(TARGET_LOCATION) / Path(key) / "raw.mp4"),
            "--output",
            str(tmp_dir),
            "--intr",
            str(tmp_dir / "intrinsic.txt"),
        ], check=True, cwd=Path(__file__).resolve().parent)

        matrices = []
        for pose_file in sorted((tmp_dir / "poses").glob("*.txt")):
            mat = np.loadtxt(pose_file)
            matrices.append(mat.tolist())
        (Path(TARGET_LOCATION) / Path(key) / "_processor_cam_slam_traj.json").write_text(json.dumps(matrices))
    finally:
        subprocess.run([
            "tar",
            "czvf",
            str(tmp_dir) + ".tgz",
            str(tmp_dir),
        ], check=True, cwd=Path(__file__).resolve().parent)
        shutil.move(str(tmp_dir) + ".tgz", Path(TARGET_LOCATION) / Path(key) / "_processor_cam_slam_output.tgz")
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    from sys import argv

    if len(argv) < 2:
        raise SystemExit("Usage: python cam_slam.py <key>")
    cam_slam(argv[1])