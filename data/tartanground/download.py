"""Downloads ANYmal TartanGround trajectories (rosbag + IMU + meta) via the tartanair package.

Requires a separate environment with `tartanairpy` installed (see the TartanGround section
of the top-level README) -- it pulls in dependencies this project's main environment doesn't need.
"""

import os
import shutil

import tartanair as ta

# For a first run, try a single environment/trajectory (smaller download, easier to debug),
# e.g. PRIORITY_CANDIDATES = ["ForestEnv"], TARGET_TRAJ = ["P2000"].
PRIORITY_CANDIDATES = [
    "ForestEnv", "GreatMarsh", "OldTownSummer", "Downtown", "Gascola",
    "ModernCityDowntown", "ModularNeighborhood", "NordicHarbor",
    "OldTownFall", "SeasonalForestAutumn", "SeasonalForestSpring", "SeasonalForestWinter",
]
TARGET_TRAJ: list[str] = []  # empty = all available trajectories
TARGET_VERSION = ["anymal"]
TARGET_MODALITIES = ["rosbag", "imu", "meta"]
DOWNLOAD_ROOT = "./anymal"


def download_data() -> None:
    ta.init(DOWNLOAD_ROOT)
    results: dict[str, list[str]] = {}

    for env in PRIORITY_CANDIDATES:
        print(f"[download] {env} ...")
        target_dir = os.path.join(DOWNLOAD_ROOT, env)
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)

        try:
            ta.download_ground(
                env=[env], version=TARGET_VERSION, traj=TARGET_TRAJ, modality=TARGET_MODALITIES, unzip=True
            )
        except (IndexError, ValueError, KeyError):
            print(f"[download] {env}: not available for this trajectory/version")
            continue
        except Exception as e:
            print(f"[download] {env}: error: {e}")
            continue

        found_files = []
        if os.path.exists(target_dir):
            # remove zips that already extracted successfully
            for root, _, files in os.walk(target_dir):
                for f in files:
                    if f.endswith(".zip") and os.path.exists(os.path.join(root, f[:-4])):
                        os.remove(os.path.join(root, f))
            for root, _, files in os.walk(target_dir):
                for f in files:
                    found_files.append(os.path.relpath(os.path.join(root, f), target_dir))

        if found_files:
            results[env] = found_files
            print(f"[download] {env}: {len(found_files)} files")
        else:
            print(f"[download] {env}: no files retrieved")

    print("\n--- availability summary ---")
    if not results:
        print("no data found for the requested trajectory in any candidate environment")
        return
    for env, files in results.items():
        has_bag = any("bag" in f for f in files)
        status = "complete" if has_bag else "partial (no bag)"
        print(f"{env}: {status}, {len(files)} files")


if __name__ == "__main__":
    download_data()
