# Part One of the SAM-IAM Pipeline
This part of the pipeline handles the segmentation of the input images and video. Given an input image, video, and points for segmentation, it returns square cropped segmentation masks for the image and each frame of the video.


## How to use

### Config
There are two config variables `sam_enabled` and `tkinter_enabled`. If `sam_enabled` is True, then more setup steps need be run:

`!mkdir -p ../checkpoints/`

`!wget -P ../checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt`

This should allow SAM2 to run properly. If `sam_enabled` is false, then segmentation will be replaced by Canny Edge detection.

`tkinter_enabled` allows the GUI to run. If set to false, it will run based on the `config.toml` file. See example file.
