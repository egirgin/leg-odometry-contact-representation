# Learning Contact Representation for Leg Odometry

Code for the paper *"Learning Contact Representation for Leg Odometry"* (CoRL'26):
self-supervised DAE+GMM and unsupervised HMM+GMM contact detectors for ZUPT-based leg odometry,
using only joint encoders + IMU (no foot force sensor).

## Install

```bash
conda create -n leg-odometry python=3.10 -y
conda activate leg-odometry
pip install -r requirements.txt
```

## Run

```bash
python main.py --config config/tartanground_hmm_gmm.yaml
python main.py --config config/tartanground_dae_gmm.yaml
python main.py --config config/ocelot_hmm_gmm.yaml
python main.py --config config/ocelot_dae_gmm.yaml
```

A sample sequence ships for each dataset (`data/tartanground/processed/ForestEnv/P2000/`,
`data/ocelot/rock/forward/`), so all four configs above run immediately after cloning. Each writes
to `output/output_<run.name>/...`: EKF history CSV, `plots/evaluation_metrics.csv`
(ATE/AHE/RPE/FPE/Frechet), and `plots/trajectory.png`.

For more TartanGround sequences, see [`data/tartanground/README.md`](data/tartanground/README.md).
For OCELOT, `dataset.kind: ocelot` expects a directory with `lowstate.csv`
(+ optional `groundtruth.csv`). **TODO: full real-world OCELOT dataset — to be added.**

## Train your own checkpoints

```bash
python -m leg_odom.features.precompute_contact_instants \
  --config leg_odom/features/default_precompute_config.yaml   # or ocelot_precompute_config.yaml

python -m leg_odom.training.gmm.train_gmm \
  --precomputed-root leg_odom/features/precomputed_tartanground \
  --output pretrained/tartanground/gmm/kinematic_weights.npz

python -m leg_odom.training.autoencoder.train_autoencoder \
  --config leg_odom/training/autoencoder/default_autoencoder_config.yaml
```

## Citation

```bibtex
@article{girgin2026contact,
  title   = {Learning Contact Representation for Leg Odometry},
  author  = {Girgin, Emre and Kilic, Cagri},
  journal = {arXiv preprint arXiv:2606.05501},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
