# Test assignment on pleural effusion segmentation

For the ease of the representation it is recommended to use a docker image.

This is a beseline on 2D dicom segmentation. As the raw data is 3D, it is decoded and converted to 2D by making horizontal slices (source code in ./root/preprocess.py). 

The raw data shall be stored in ./raw_data by default, the preprocessing operation will put .npy images and masks to the raw_data dir.

All experiment params and settings are stored in the .ini file (the example is provided in ./root/default_params.ini).

All experiment artefacts (model weights, images, mask overlays etc.) are stored in ./artefacts

The baseline is integrated with wandb, so it is required to pass the api key to the .ini file.

The training and inference code is provided in ./root/

The requirements.txt are created considering the work inside the container so they do not include common libraries like pytorch, numpy etc.

P.S. As I currently do not have access to gpu, it is difficult to test the baselint as it is. The code that is confirmed to work is saved at ./kaggle/pleural-2d-seg.ipynb