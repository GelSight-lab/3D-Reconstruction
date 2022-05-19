# 3D-Reconstruction

Poisson reconstruction for any curved surfaces and fisheye cameras. The repo contains a few examples with synthetic data, however it works well on experimental data. Examples with experimental data will be published in a few days.


## Features

- Reconstruction on any surface based on fisheye camera projection
- Reconstruction using subtracted poisson equation
- DST-based Fast prossion solver
- GPU version based on torch
## To do
- [ ] Establish initial geometry correspondacne
- [ ] Contact region detecting to decrease the size of the problem
- [ ] Error analysis
- [ ] User-defined fisheye (or other camera) projection
