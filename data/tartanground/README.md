# TartanGround download & processing

Pulls ANYmal trajectories from [TartanAir](https://tartanair.org) via the `tartanair` package
and converts them into the `imu.csv` + `*_bag.csv` layout the main repo's `TartangroundDataset`
expects.

Uses a separate environment — `tartanair`'s dependencies aren't needed anywhere else in this repo.

## Setup

```bash
conda create -n tartan python=3.9 -y
conda activate tartan

git clone https://github.com/castacks/tartanairpy.git
pip install -e ./tartanairpy
pip install rosbags pandas scipy
```

## Run (from this directory)

```bash
python download.py      # -> ./anymal/<ENV>/Data_anymal/<TRAJ>/{rosbags,imu,meta}
python process_bag.py   # -> ./processed/<ENV>/<TRAJ>/<BAG_STEM>_bag.csv
python process_imu.py   # -> ./processed/<ENV>/<TRAJ>/imu.csv
```

Edit `PRIORITY_CANDIDATES` / `TARGET_TRAJ` at the top of `download.py` to pick which
environments/trajectories to fetch (default: all trajectories across a list of environments).

Then point `dataset.sequence_dir` in an experiment config at one `processed/<ENV>/<TRAJ>/` folder.

## Notes

- On Apple Silicon, `tartanairpy` fails to import (`SixPlanarNumba` / numba is hard-coded to
  x86_64-only, and `cupy` has no CPU build). Both are only used for image/depth resampling we
  don't need here. Workarounds: make the `SixPlanarNumba` import in `tartanairpy/tartanair/customizer.py`
  optional (wrap in `try/except ImportError`), and drop `cupy` from `pyproject.toml`.
- `tartanairpy` pulls in `huggingface_hub`; if downloads fail with `min() arg is an empty sequence`,
  pin an older `huggingface_hub` (e.g. `pip install "huggingface_hub==0.26.5" requests`).
