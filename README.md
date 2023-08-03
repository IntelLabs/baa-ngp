#  BAA:sheep:-NGP: Bundle-Adjusting Accelerated Neural Graphics Primitives
This repository contains the official Implementation for "BAA-NGP: Bundle-Adjusting Accelerated Neural Graphics Primitives".

# Installation
Tested on `NVIDIA A100` and `NVIDIA RTX3090`.
### Dependencies
- python >= 3.8
- [pytorch](https://pytorch.org/get-started/locally/) >= 2.0.1
- [tinycudann](https://github.com/NVlabs/tiny-cuda-nn) >= 1.7
- [nerfacc](https://github.com/KAIR-BAIR/nerfacc) >= 0.5.0
- Install the remaining dependencies via `pip install -r requirements.txt`

# Experiments

## Blender/nerf_synthetic dataset

- Download nerf_synthetic (1.6G) from [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1). 
    - 100 train images
    - 200 test images

- Train and run
    ```
    python baangp/train_baangp.py --scene [lego] --data-root [your_data_root] --save-dir [your_save_dir] --c2f 0.1 0.5
    ```
## LLFF dataset

Coming soon.

# Acknowledgements

`BAA-NGP` code is heavily based on [nerfacc](https://github.com/KAIR-BAIR/nerfacc) and [barf](https://github.com/chenhsuanlin/bundle-adjusting-NeRF).

# Citation
If you use this code for your research, please cite our paper [BAA-NGP: Bundle-Adjusting Accelerated Neural Graphics Primitives](http://arxiv.org/abs/2306.04166)

```bibtex
@article{liu2023baangp,
  title={BAA-NGP: Bundle-Adjusting Accelerated Neural Graphics Primitives.},
  author={Sainan Liu and Shan Lin and Jingpei Lu and Shreya Saha and Alexey Supikov and Michael Yip},
  journal={arXiv preprint arXiv:2306.04166},
  year={2023}
}
```