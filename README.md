# open_clip_3d
Simply modified open_clip to work on 3d data samples, originally at [open_clip_python](https://github.com/mlfoundations/open_clip). Least modification to adapt it to 3D training as you can now add an option `dims=3` in the python file `open_clip/modified_resnet.py` to train on 3D data. (I never bothered to change the ViT case)
