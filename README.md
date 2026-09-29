# OSSD: LiDAR-Language Object Tracking via Orthogonal Slot Disentanglement

## Dependencies

We list the most important dependencies below.

| Dependency | Version |
| :--- | :--- |
| python | 3.8.0 |
| pytorch | 1.8.0 (cuda11.1, cudnn8.0.5) |
| pytorch-lightning | 1.5.10 |
| pytorch3d | 0.6.2 |
| open3d | 0.15.2 |
| shapely | 1.8.1 |
| torchvision | 0.9.0 |

Other dependencies can be found in `requirements.txt`.

## Datasets

### Original Datasets

- **GSOT3D**: Download from the [official GSOT3D page](https://huggingface.co/datasets/Ailovejinx/GSOT3D).
- **NuScenes**: Download from the [official NuScenes download page](https://www.nuscenes.org/download).

### Language Annotations

We provide the natural language annotations generated in this work for GSOT3D and NuScenes.

- **GSOT3D language annotations**: [download link]
- **NuScenes language annotations**: [download link]

The annotations are currently under embargo and will be made publicly available upon publication of this article. Reviewers can access them via a private link provided in the cover letter.

## Quick Start

### Training
To train a model, you must specify the .yaml file. The .yaml file contains all the configurations of the dataset and the model. We provide .yaml files under the configs/ directory.

Note: Before running the code, you will need to edit the .yaml file by setting the data_root_dir argument as the correct root of the dataset.

```bash
python main.py configs/newmbpsem_gsot3d.yaml --gpus 0 1
```

## Acknowledgement

This repo is heavily built upon Open3DSOT, MBPTrack, and DAPT.
We thank the authors for their excellent work.
