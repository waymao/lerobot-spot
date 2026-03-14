#!/usr/bin/env python3
"""
Simple keyboard teleop + LeRobot dataset recorder for SpotRobot (pose target).

This records trainable LeRobot episodes while you drive Spot with keyboard commands.
Arm control uses Cartesian pose target keys:
  arm.pose.x / arm.pose.y / arm.pose.z / arm.pose.qw / arm.pose.qx / arm.pose.qy / arm.pose.qz

Controls (press repeatedly, commands are incremental):
  Base:
    w/s : increase/decrease forward velocity (base.vx)
    z/c : increase/decrease lateral velocity (base.vy)
    a/d : increase/decrease yaw rate (base.vyaw)
    space: zero all base velocities

  Arm translation:
    u/j : x +/-
    i/k : y +/-
    o/l : z +/-
    y   : x- y+ (diagonal)
    h   : x+ y- (diagonal)

  Arm orientation:
    7/4 : roll +/-
    8/5 : pitch +/-
    9/6 : yaw +/-
    r   : reset arm target to current observed pose
    p   : reset arm target to original startup pose

  Gripper:
    g   : open gripper (100%)
    b   : half-open gripper (50%)
    t   : close gripper (0%)

  Episode/session:
    n : end current episode early (prompts save/discard)
    q : quit (prompts save/discard for current episode)
    h : print help
"""

from __future__ import annotations

import argparse
import math
import select
import sys
import termios
import time
import tty
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts, hw_to_dataset_features
from lerobot.processor import make_default_processors
from lerobot.utils.constants import ACTION, OBS_STR

from lerobot_robot_spot import SpotRobot, SpotRobotConfig

POSE_KEYS = (
    "arm.pose.x",
    "arm.pose.y",
    "arm.pose.z",
    "arm.pose.qw",
    "arm.pose.qx",
    "arm.pose.qy",
    "arm.pose.qz",
)


@dataclass
class TeleopState:
    vx: float = 0.0
    vy: float = 0.0
    vyaw: float = 0.0
    gripper_pct: float = -1.0  # -1 = no command yet (backward compat), 0–100 = open percentage
    end_episode: bool = False
    quit_all: bool = False

    def zero_base(self) -> None:
        self.vx = 0.0
        self.vy = 0.0
        self.vyaw = 0.0


@contextmanager
def raw_stdin():
    if not sys.stdin.isatty():
        raise RuntimeError("This script requires an interactive TTY terminal.")
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def clamp(value: float, limit: float) -> float:
    return float(max(-limit, min(limit, value)))


def normalize_quat(qw: float, qx: float, qy: float, qz: float) -> tuple[float, float, float, float]:
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n < 1e-8:
        return 1.0, 0.0, 0.0, 0.0
    return qw / n, qx / n, qy / n, qz / n


def quat_mul(
    q1: tuple[float, float, float, float], q2: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def delta_rpy_to_quat(dr: float, dp: float, dy: float) -> tuple[float, float, float, float]:
    cr = math.cos(dr * 0.5)
    sr = math.sin(dr * 0.5)
    cp = math.cos(dp * 0.5)
    sp = math.sin(dp * 0.5)
    cy = math.cos(dy * 0.5)
    sy = math.sin(dy * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return normalize_quat(qw, qx, qy, qz)


def apply_rpy_delta(target: dict[str, float], dr: float, dp: float, dy: float) -> None:
    q_cur = (
        float(target["arm.pose.qw"]),
        float(target["arm.pose.qx"]),
        float(target["arm.pose.qy"]),
        float(target["arm.pose.qz"]),
    )
    q_delta = delta_rpy_to_quat(dr, dp, dy)
    q_new = quat_mul(q_delta, q_cur)
    qw, qx, qy, qz = normalize_quat(*q_new)
    target["arm.pose.qw"] = qw
    target["arm.pose.qx"] = qx
    target["arm.pose.qy"] = qy
    target["arm.pose.qz"] = qz


def poll_keys() -> list[str]:
    pressed: list[str] = []
    while True:
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready:
            break
        ch = sys.stdin.read(1)
        if ch:
            pressed.append(ch)
    return pressed


def wait_for_save_or_discard() -> bool:
    """Ask user whether to save or discard the just-finished episode.

    Returns True if the episode should be saved, False to discard.
    """
    _ = poll_keys()
    print("\nSave or discard this episode? [s=save / d=discard]: ", end="", flush=True)
    while True:
        ready, _, _ = select.select([sys.stdin], [], [], 0.1)
        if not ready:
            continue
        ch = sys.stdin.read(1)
        if ch.lower() == "s":
            print("save")
            return True
        if ch.lower() == "d":
            print("discard")
            return False


def wait_for_manual_reset() -> bool:
    """Wait until Enter to continue; return False if user chooses to quit."""
    # Drain buffered keys from teleop loop first.
    _ = poll_keys()
    print("Reset scene, then press Enter for next episode (or press q then Enter to stop).")
    while True:
        ready, _, _ = select.select([sys.stdin], [], [], 0.1)
        if not ready:
            continue
        ch = sys.stdin.read(1)
        if ch in ("\n", "\r"):
            return True
        if ch.lower() == "q":
            # Consume optional trailing newline.
            return False


def print_help() -> None:
    print("\nKeyboard teleop (pose target)")
    print("  base: w/s vx, z/c vy, a/d vyaw, <space> stop base")
    print("  pos : u/j x, i/k y, o/l z, y x-y+, h x+y-")
    print("  rot : 7/4 roll, 8/5 pitch, 9/6 yaw")
    print("  grip: g open(100%), b half(50%), t close(0%)")
    print("  arm : r reset target to current observed hand pose, p reset to startup pose")
    print("  run : n end episode (prompts save/discard), q quit, h help")


def reset_pose_target_from_obs(arm_target: dict[str, float], obs: dict[str, Any]) -> None:
    for k in POSE_KEYS:
        arm_target[k] = float(obs.get(k, arm_target.get(k, 0.0)))


def hold_base_with_arm_pose(
    robot: SpotRobot, arm_pose: dict[str, float], repeats: int, rate_hz: float
) -> None:
    """Send repeated zero-base + arm-pose commands to reinforce a reset move."""
    steps = max(1, int(repeats))
    sleep_s = 1.0 / max(1.0, rate_hz)
    action = {"base.vx": 0.0, "base.vy": 0.0, "base.vyaw": 0.0, **arm_pose}
    for _ in range(steps):
        robot.send_action(action)
        time.sleep(sleep_s)


def update_state_from_keys(
    keys: list[str],
    state: TeleopState,
    arm_target: dict[str, float],
    original_arm_pose: dict[str, float],
    obs: dict[str, Any],
    base_step_vx: float,
    base_step_vy: float,
    base_step_vyaw: float,
    arm_step_xyz: float,
    arm_step_rpy: float,
    max_vx: float,
    max_vy: float,
    max_vyaw: float,
) -> None:
    for raw_ch in keys:
        ch = raw_ch.lower()
        if ch == "w":
            state.vx = clamp(state.vx + base_step_vx, max_vx)
        elif ch == "s":
            state.vx = clamp(state.vx - base_step_vx, max_vx)
        elif ch == "z":
            state.vy = clamp(state.vy + base_step_vy, max_vy)
        elif ch == "c":
            state.vy = clamp(state.vy - base_step_vy, max_vy)
        elif ch == "a":
            state.vyaw = clamp(state.vyaw + base_step_vyaw, max_vyaw)
        elif ch == "d":
            state.vyaw = clamp(state.vyaw - base_step_vyaw, max_vyaw)
        elif ch == " ":
            state.zero_base()
        elif ch == "u":
            arm_target["arm.pose.x"] += arm_step_xyz
        elif ch == "j":
            arm_target["arm.pose.x"] -= arm_step_xyz
        elif ch == "i":
            arm_target["arm.pose.y"] += arm_step_xyz
        elif ch == "k":
            arm_target["arm.pose.y"] -= arm_step_xyz
        elif ch == "y":
            arm_target["arm.pose.x"] -= arm_step_xyz
            arm_target["arm.pose.y"] += arm_step_xyz
        elif ch == "h":
            arm_target["arm.pose.x"] += arm_step_xyz
            arm_target["arm.pose.y"] -= arm_step_xyz
        elif ch == "o":
            arm_target["arm.pose.z"] += arm_step_xyz
        elif ch == "l":
            arm_target["arm.pose.z"] -= arm_step_xyz
        elif ch == "7":
            apply_rpy_delta(arm_target, +arm_step_rpy, 0.0, 0.0)
        elif ch == "4":
            apply_rpy_delta(arm_target, -arm_step_rpy, 0.0, 0.0)
        elif ch == "8":
            apply_rpy_delta(arm_target, 0.0, +arm_step_rpy, 0.0)
        elif ch == "5":
            apply_rpy_delta(arm_target, 0.0, -arm_step_rpy, 0.0)
        elif ch == "9":
            apply_rpy_delta(arm_target, 0.0, 0.0, +arm_step_rpy)
        elif ch == "6":
            apply_rpy_delta(arm_target, 0.0, 0.0, -arm_step_rpy)
        elif ch == "g":
            state.gripper_pct = 100.0
        elif ch == "b":
            state.gripper_pct = 50.0
        elif ch == "t":
            state.gripper_pct = 0.0
        elif ch == "r":
            reset_pose_target_from_obs(arm_target, obs)
        elif ch == "p":
            arm_target.update(original_arm_pose)
        elif ch == "n":
            state.end_episode = True
        elif ch == "q":
            state.quit_all = True
            state.end_episode = True
        elif ch == "h":
            print_help()


def make_dataset(
    robot: SpotRobot,
    repo_id: str,
    root: str | None,
    fps: int,
    use_videos: bool,
    store_cameras: bool = True,
    camera_shapes: dict[str, tuple[int, int, int]] | None = None,
    resume: bool = False,
) -> LeRobotDataset:
    teleop_action_processor, _, robot_observation_processor = make_default_processors()
    obs_features = dict(robot.observation_features)
    if store_cameras and camera_shapes:
        for key, shape in camera_shapes.items():
            if key in obs_features and isinstance(obs_features[key], tuple):
                obs_features[key] = shape

    features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=use_videos,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=obs_features),
            use_videos=use_videos,
        ),
    )
    if store_cameras and not use_videos:
        # Keep camera frames as image features when video encoding is disabled.
        camera_features = {k: v for k, v in obs_features.items() if isinstance(v, tuple)}
        if camera_features:
            features = combine_feature_dicts(
                features,
                hw_to_dataset_features(camera_features, OBS_STR, use_video=False),
            )
    resolved_root: Path | None = None
    if root is not None:
        base = Path(root).expanduser()
        # Treat --dataset-root as a parent directory and place dataset under repo_id.
        resolved_root = base / repo_id

    if resume:
        if resolved_root is None:
            return LeRobotDataset(repo_id)
        return LeRobotDataset(repo_id, root=resolved_root)

    if resolved_root is not None and resolved_root.exists():
        # Avoid FileExistsError from LeRobotDataset.create(...) which expects a new root.
        stamp = time.strftime("%Y%m%d_%H%M%S")
        resolved_root = Path(str(resolved_root) + f"_{stamp}")

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=resolved_root,
        robot_type=robot.name,
        use_videos=use_videos,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Spot keyboard teleop recorder (pose target)")
    p.add_argument("--hostname", required=True)
    p.add_argument("--username", required=True)
    p.add_argument("--password", required=True)
    p.add_argument(
        "--image-sources",
        nargs="+",
        default=["frontleft_fisheye_image", "frontright_fisheye_image", "hand_color_image"],
    )
    p.add_argument("--image-width", type=int, default=640)
    p.add_argument("--image-height", type=int, default=480)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument("--episode-time-s", type=float, default=30.0)
    p.add_argument(
        "--manual-reset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pause between episodes and wait for Enter (default: true).",
    )
    p.add_argument(
        "--reset-time-s",
        type=float,
        default=10.0,
        help="Reset wait duration when --no-manual-reset is used.",
    )
    p.add_argument("--task", type=str, default="teleop spot pose task")
    p.add_argument("--repo-id", type=str, required=True, help="e.g. user/spot-pose-demo")
    p.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Parent directory for datasets. Final path becomes <dataset-root>/<repo-id>.",
    )
    p.add_argument("--resume", action="store_true", help="Append episodes to an existing dataset.")
    p.add_argument("--video", action="store_true", help="Store camera frames as encoded videos")
    p.add_argument(
        "--store-cameras",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store camera frames in dataset (as videos with --video, otherwise as images).",
    )

    p.add_argument("--base-step-vx", type=float, default=0.05)
    p.add_argument("--base-step-vy", type=float, default=0.05)
    p.add_argument("--base-step-vyaw", type=float, default=0.08)
    p.add_argument("--arm-step-xyz", type=float, default=0.04)
    p.add_argument("--arm-step-rpy-deg", type=float, default=10.0)
    p.add_argument("--max-vx", type=float, default=0.35)
    p.add_argument("--max-vy", type=float, default=0.20)
    p.add_argument("--max-vyaw", type=float, default=0.50)
    p.add_argument(
        "--power-off-on-exit",
        action="store_true",
        help="If set, calls robot.disconnect() (power off). Default keeps Spot powered.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = SpotRobotConfig(
        hostname=args.hostname,
        username=args.username,
        password=args.password,
        image_sources=args.image_sources,
        image_width=args.image_width,
        image_height=args.image_height,
    )
    robot = SpotRobot(cfg)

    print("Connecting to Spot...")
    robot.connect()
    print(f"Connected={robot.is_connected} calibrated={robot.is_calibrated}")
    startup_obs = robot.get_observation()
    camera_shapes: dict[str, tuple[int, int, int]] = {}
    for key, spec in robot.observation_features.items():
        if not isinstance(spec, tuple):
            continue
        value = startup_obs.get(key)
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) == 3:
            camera_shapes[key] = (int(shape[0]), int(shape[1]), int(shape[2]))

    dataset = make_dataset(
        robot=robot,
        repo_id=args.repo_id,
        root=args.dataset_root,
        fps=args.fps,
        use_videos=args.video,
        store_cameras=args.store_cameras,
        camera_shapes=camera_shapes,
        resume=args.resume,
    )
    print(f"Dataset root: {dataset.root}")
    print(f"Camera storage: {'video' if args.video and args.store_cameras else ('image' if args.store_cameras else 'disabled')}")

    state = TeleopState()
    loop_dt = 1.0 / float(args.fps)
    arm_step_rpy = math.radians(args.arm_step_rpy_deg)

    print_help()

    try:
        original_arm_pose = {k: float(startup_obs.get(k, 0.0)) for k in POSE_KEYS}

        with raw_stdin():
            saved_episodes = 0
            while saved_episodes < args.num_episodes:
                if state.quit_all:
                    break

                print(f"\nEpisode {saved_episodes + 1}/{args.num_episodes} start")
                episode_t0 = time.perf_counter()
                next_tick = episode_t0
                status_t = 0.0
                state.end_episode = False

                obs = robot.get_observation()
                arm_target = {k: float(obs.get(k, 0.0)) for k in POSE_KEYS}
                state.zero_base()

                while not state.end_episode:
                    now = time.perf_counter()
                    if now - episode_t0 >= args.episode_time_s:
                        break

                    keys = poll_keys()
                    if keys:
                        update_state_from_keys(
                            keys=keys,
                            state=state,
                            arm_target=arm_target,
                            original_arm_pose=original_arm_pose,
                            obs=obs,
                            base_step_vx=args.base_step_vx,
                            base_step_vy=args.base_step_vy,
                            base_step_vyaw=args.base_step_vyaw,
                            arm_step_xyz=args.arm_step_xyz,
                            arm_step_rpy=arm_step_rpy,
                            max_vx=args.max_vx,
                            max_vy=args.max_vy,
                            max_vyaw=args.max_vyaw,
                        )

                    obs = robot.get_observation()
                    action = {
                        "base.vx": state.vx,
                        "base.vy": state.vy,
                        "base.vyaw": state.vyaw,
                        **arm_target,
                        "arm.gripper_open_percentage": state.gripper_pct,
                    }
                    sent = robot.send_action(action)

                    frame = combine_feature_dicts(
                        build_dataset_frame(dataset.features, obs, OBS_STR),
                        build_dataset_frame(dataset.features, sent, ACTION),
                        {
                            "task": args.task,
                        },
                    )
                    dataset.add_frame(frame)

                    if now - status_t > 0.5:
                        obs_x = float(obs.get("arm.pose.x", 0.0))
                        obs_y = float(obs.get("arm.pose.y", 0.0))
                        obs_z = float(obs.get("arm.pose.z", 0.0))
                        obs_qw = float(obs.get("arm.pose.qw", 1.0))
                        obs_qx = float(obs.get("arm.pose.qx", 0.0))
                        obs_qy = float(obs.get("arm.pose.qy", 0.0))
                        obs_qz = float(obs.get("arm.pose.qz", 0.0))
                        obs_grip = float(obs.get("arm.gripper_open_percentage", 0.0))
                        tgt_x = float(arm_target.get("arm.pose.x", 0.0))
                        tgt_y = float(arm_target.get("arm.pose.y", 0.0))
                        tgt_z = float(arm_target.get("arm.pose.z", 0.0))
                        tgt_qw = float(arm_target.get("arm.pose.qw", 1.0))
                        tgt_qx = float(arm_target.get("arm.pose.qx", 0.0))
                        tgt_qy = float(arm_target.get("arm.pose.qy", 0.0))
                        tgt_qz = float(arm_target.get("arm.pose.qz", 0.0))
                        grip_str = f"{state.gripper_pct:.0f}%" if state.gripper_pct >= 0.0 else "—"
                        print(
                            f"\r vx={state.vx:+.2f} vy={state.vy:+.2f} vyaw={state.vyaw:+.2f} "
                            f"ee_xyz=({obs_x:+.3f},{obs_y:+.3f},{obs_z:+.3f}) "
                            f"ee_quat=({obs_qw:+.3f},{obs_qx:+.3f},{obs_qy:+.3f},{obs_qz:+.3f}) "
                            f"grip=obs:{obs_grip:.0f}%/cmd:{grip_str} "
                            f"target_xyz=({tgt_x:+.3f},{tgt_y:+.3f},{tgt_z:+.3f}) "
                            f"target_quat=({tgt_qw:+.3f},{tgt_qx:+.3f},{tgt_qy:+.3f},{tgt_qz:+.3f}) "
                            f"frames={dataset.episode_buffer['size']}",
                            end="",
                            flush=True,
                        )
                        status_t = now

                    next_tick += loop_dt
                    sleep_s = next_tick - time.perf_counter()
                    if sleep_s > 0:
                        time.sleep(sleep_s)

                state.zero_base()
                robot.send_action(
                    {
                        "base.vx": 0.0,
                        "base.vy": 0.0,
                        "base.vyaw": 0.0,
                        **arm_target,
                    }
                )

                if wait_for_save_or_discard():
                    print("Saving episode...")
                    dataset.save_episode()
                    saved_episodes += 1
                    print(f"Saved. total_episodes={dataset.num_episodes} total_frames={dataset.num_frames}")
                else:
                    dataset.clear_episode_buffer()
                    print("Episode discarded.")

                if state.quit_all or saved_episodes >= args.num_episodes:
                    continue

                print("Opening gripper before reset...")
                robot.send_action(
                    {"base.vx": 0.0, "base.vy": 0.0, "base.vyaw": 0.0,
                     **arm_target, "arm.gripper_open_percentage": 100.0}
                )
                time.sleep(1.0)

                print("Resetting arm to original startup pose...")
                hold_base_with_arm_pose(
                    robot=robot,
                    arm_pose=original_arm_pose,
                    repeats=50,
                    rate_hz=float(args.fps),
                )

                print("Closing gripper after reset...")
                robot.send_action(
                    {"base.vx": 0.0, "base.vy": 0.0, "base.vyaw": 0.0,
                     **original_arm_pose, "arm.gripper_open_percentage": 0.0}
                )
                time.sleep(1.0)
                state.gripper_pct = 0.0

                if args.manual_reset:
                    if not wait_for_manual_reset():
                        state.quit_all = True
                else:
                    print(f"Resetting scene for {args.reset_time_s:.1f}s...")
                    time.sleep(max(0.0, args.reset_time_s))

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        print("\nStopping base motion...")
        try:
            obs = robot.get_observation()
            hold = {"base.vx": 0.0, "base.vy": 0.0, "base.vyaw": 0.0}
            hold.update({k: float(obs.get(k, 0.0)) for k in POSE_KEYS})
            robot.send_action(hold)
        except Exception:
            pass

        if args.power_off_on_exit:
            print("Disconnecting (power off)...")
            robot.disconnect()
        else:
            print("Disconnecting (keep powered)...")
            robot.disconnect_keep_powered()
        print("Done.")


if __name__ == "__main__":
    main()
