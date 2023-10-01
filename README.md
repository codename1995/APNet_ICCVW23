# APNet: Urban-level Scene Segmentation of Aerial Images and Point Clouds

**[Paper (To be updated)]()** **|** **[Poster](https://amsuni-my.sharepoint.com/:b:/g/personal/w_wei2_uva_nl/Ecoj28p8T2FFmI6QRQZdhUABWclgn4lO9hRLu84Dez7tvw?e=CUbWbV)** **|** **[Slides](https://amsuni-my.sharepoint.com/:b:/g/personal/w_wei2_uva_nl/EahGmaCDG0dDs4w9vT1jRXcBhbigtEPsnnaTiJ8BvOGxNg?e=4t1cX6)**

This repo is the official implementation for the ICCVW'23 paper: APNet: Urban-level Scene Segmentation of Aerial Images and Point Clouds

## Dependencies

To run our code first install the dependencies with:

```
conda env create -f environment.yaml
```

## Dataset
### SensatUrban Pre-processing

The code will be released after cleaning.

## Running the code

Then run the following command:
```
sh run_files/train_eval.sh
```

# Citation

If you use this repo, please cite as :

```
@inproceedings{wei2023apnet,
    author = {Weijie Wei and Martin R. Oswald and Fatemeh Karimi Nejadasl and Theo Gevers},
    title = {{APNet: Urban-level Scene Segmentation of Aerial Images and Point Clouds}},
    booktitle = {{Proceedings of the IEEE International Conference on Computer Vision Workshops (ICCVW)}},
    year = {2023}
}
```

# Acknowledgement
Our code is heavily inspired by the following projects:
1. RandLA-Net: https://github.com/QingyongHu/RandLA-Net
2. RandLA-Net-pytorch: https://github.com/tsunghan-wu/RandLA-Net-pytorch
3. HRNet: https://github.com/HRNet/HRNet-Semantic-Segmentation
4. KPConv: https://github.com/HuguesTHOMAS/KPConv-PyTorch
5. KPRNet: https://github.com/DeyvidKochanov-TomTom/kprnet
6. SensatUrban-BEV-Seg3D: https://github.com/zouzhenhong98/SensatUrban-BEV-Seg3D

Thanks for their contributions.
