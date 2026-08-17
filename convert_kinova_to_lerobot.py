#!/usr/bin/env python3
"""
Convert Kinova Gen3 HDF5 episodes -> LeRobot v2.0 dataset for openpi.

Run in the OPENPI venv (needs lerobot 0.1.0), not kinova_env:

    cd ~/openpi
    uv run python convert_kinova_to_lerobot.py --data-dir ~/kinova_collect/episodes

Output lands in $HF_LEROBOT_HOME (default ~/.cache/huggingface/lerobot).

WHAT GOES IN THE DATASET
------------------------
state[t]   (8,) = [j0..j6 measured joint positions (rad), measured gripper]
actions[t] (8,) = [j0..j6 joint positions at t+1 (rad), commanded gripper at t]

Actions are ABSOLUTE joint positions. LeRobotKinovaDataConfig converts them to
deltas at training time via DeltaActions(make_bool_mask(7, -1)) -- 7 joints
delta, gripper absolute. Do NOT also make them deltas here; applying the
conversion twice is a silent, hard-to-diagnose bug.

Gripper: state uses MEASURED (plateaus ~0.72 when the block blocks the fingers),
action uses COMMANDED (spans the full 0..1). This mirrors how DROID and LIBERO
separate proprioception from control signal.
"""

import argparse
import glob
import os
import shutil

import cv2
import h5py
import numpy as np
import tyro

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

TASK = "pick up the red block and put it in the white bowl"
IMG_SIZE = 256          # LIBERO uses 256; openpi resizes to 224 at train time
FPS = 15

# No-op thresholds -- must match what the audit script used
JOINT_EPS = 0.002       # rad
GRIP_EPS = 0.005


def resize_frame(img, size=IMG_SIZE, center_crop=False):
    """640x480 BGR -> size x size RGB.

    center_crop=False (default) squashes the full 4:3 frame into a square,
    preserving the whole workspace you framed. center_crop=True keeps the
    aspect ratio but throws away the left and right edges.
    """
    if center_crop:
        h, w = img.shape[:2]
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        img = img[y0:y0 + s, x0:x0 + s]
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main(
    data_dir: str = "~/kinova_collect/episodes",
    repo_name: str = "thithikhine/kinova_gen3_pick_place",
    center_crop: bool = False,
    filter_noops: bool = True,
    push_to_hub: bool = False,
    dry_run: bool = False,
):
    data_dir = os.path.expanduser(data_dir)
    files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))
    if not files:
        raise SystemExit(f"No .hdf5 files in {data_dir}")

    print(f"{len(files)} episodes in {data_dir}")
    print(f"task        : {TASK}")
    print(f"image       : {IMG_SIZE}x{IMG_SIZE} ({'center-crop' if center_crop else 'full frame squashed'})")
    print(f"no-op filter: {'on' if filter_noops else 'off'}")
    print(f"output      : {HF_LEROBOT_HOME / repo_name}\n")

    if dry_run:
        tot = kept = 0
        for f in files:
            with h5py.File(f) as h:
                n = int(h.attrs["num_frames"])
                jp = h["joint_pos_rad"][:]
                g = h["gripper"][:]
                keep = noop_mask(jp, g) if filter_noops else np.ones(n, bool)
                tot += n
                kept += keep.sum()
                print(f"  {os.path.basename(f)}  {n:4d} -> {keep.sum():4d}")
        print(f"\nDRY RUN: {tot} frames -> {kept} kept ({100*(tot-kept)/tot:.1f}% dropped)")
        return

    out = HF_LEROBOT_HOME / repo_name
    if out.exists():
        resp = input(f"{out} exists. Delete and rebuild? [y/N] ")
        if resp.lower() != "y":
            raise SystemExit("aborted")
        shutil.rmtree(out)

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="kinova_gen3",
        fps=FPS,
        features={
            "image": {
                "dtype": "image",
                "shape": (IMG_SIZE, IMG_SIZE, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    total_in = total_out = 0
    for path in files:
        with h5py.File(path) as h:
            n = int(h.attrs["num_frames"])
            jp = h["joint_pos_rad"][:]        # (n, 7) radians, unwrapped WITHIN episode
            jp = canonicalize_branch(jp)      # ...but not ACROSS episodes -- fix that
            grip = h["gripper"][:]            # (n,)   measured
            cmd_grip = h["cmd_gripper"][:]    # (n,)   commanded

            # state = measured joints + measured gripper
            state = np.concatenate([jp, grip[:, None]], axis=1).astype(np.float32)

            # action[t] = joints at t+1 (where the arm actually went) + commanded gripper at t
            # Last frame has no t+1, so hold position.
            nxt = np.concatenate([jp[1:], jp[-1:]], axis=0)
            actions = np.concatenate([nxt, cmd_grip[:, None]], axis=1).astype(np.float32)

            keep = noop_mask(jp, grip) if filter_noops else np.ones(n, bool)

            written = 0
            for i in range(n):
                if not keep[i]:
                    continue
                img = cv2.imdecode(h["img_front"][i], cv2.IMREAD_COLOR)
                dataset.add_frame({
                    "image": resize_frame(img, IMG_SIZE, center_crop),
                    "state": state[i],
                    "actions": actions[i],
                    "task": TASK,
                })
                written += 1

            dataset.save_episode()
            total_in += n
            total_out += written
            print(f"  {os.path.basename(path)}  {n:4d} -> {written:4d} frames")

    print(f"\n{len(files)} episodes, {total_in} frames in, {total_out} written")
    print(f"dataset: {out}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["kinova", "gen3", "pick-place"],
            private=True,
            push_videos=True,
            license="apache-2.0",
        )
        print("pushed to hub")


def canonicalize_branch(jp):
    """Put every episode on the same 2*pi branch.

    The recorder unwraps joint angles within an episode, but resets its
    reference each time. j0 and j4 sit on the 0/2*pi boundary at the home pose,
    so some episodes start near 0.00 rad and others near 6.28 rad -- physically
    identical, numerically a full turn apart. Measured spreads at episode start
    were 6.20 and 6.28 rad for j0 and j4, versus 0.00-0.14 for every other joint.

    Wrap the FIRST frame into [-pi, pi] and shift the whole trajectory by that
    same constant offset. Within-episode continuity is preserved exactly.

    NOTE FOR DEPLOYMENT: the policy client must apply the same wrap to the live
    state before sending it to the server, or training and serving disagree.
    """
    start = jp[0]
    wrapped = (start + np.pi) % (2 * np.pi) - np.pi
    return jp + (wrapped - start)


def noop_mask(jp, grip):
    """True where the frame is NOT a no-op.

    Filtering these is crucial: expressive single-step policies otherwise learn
    to imitate the no-ops and freeze indefinitely mid-rollout. Teleop data is
    full of pauses while the operator thinks.

    The first frame is always kept so every episode has a starting state.
    """
    n = len(jp)
    dj = np.abs(np.diff(jp, axis=0)).max(1)
    dg = np.abs(np.diff(grip))
    still = (dj < JOINT_EPS) & (dg < GRIP_EPS)
    keep = np.ones(n, bool)
    keep[1:] = ~still
    return keep


if __name__ == "__main__":
    tyro.cli(main)
