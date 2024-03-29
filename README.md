#  BAA-NGP: Bundle-Adjusting Accelerated Neural Graphics Primitives
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

## Disclaimer

> This “research quality code”  is for Non-Commercial purposes and provided by Intel “As Is” without any express or implied warranty of any kind. Please see the dataset's applicable license for terms and conditions. Intel does not own the rights to this data set and does not confer any rights to it. Intel does not warrant or assume responsibility for the accuracy or completeness of any information, text, graphics, links or other items within the code. A thorough security review has not been performed on this code. Additionally, this repository may contain components that are out of date or contain known security vulnerabilities.

> nerf_synthetic dataset: Please see the dataset's applicable license for terms and conditions. Intel does not own the rights to this data set and does not confer any rights to it.

## Datasets & Models Disclaimer :

> To the extent that any public datasets are referenced by Intel or accessed using tools or code on this site those datasets are provided by the third party indicated as the data source. Intel does not create the data, or datasets, and does not warrant their accuracy or quality. By accessing the public dataset(s), or using a model trained on those datasets, you agree to the terms associated with those datasets and that your use complies with the applicable license. 

> Intel expressly disclaims the accuracy, adequacy, or completeness of any public datasets, and is not liable for any errors, omissions, or defects in the data, or for any reliance on the data.  Intel is not liable for any liability or damages relating to your use of public datasets.
