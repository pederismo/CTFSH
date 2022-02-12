# CTFSH
This repository contains the code for the paper [CTFSH: Full Head CT Anomaly Detection with Unsupervised Learning](https://openreview.net/forum?id=9qZUArz732o&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DMIDL.io%2F2022%2FConference%2FAuthors%23your-submissions)).

## Requirements
All the necessary package requirements are listed in file `requirements.txt` and can be installed by running `pip install -r requirements.txt`:

```
matplotlib==3.3.4  
monai==0.7.0  
nibabel==3.2.1  
numpy==1.19.5  
pandas==1.1.5  
pytorch_lightning==1.5.2  
pytorch_msssim==0.2.1  
scikit_image==0.17.2  
scikit_learn==1.0.2
scipy==1.5.4
seaborn==0.11.2
torch==1.11.0.dev20211118+cu102
torchvision==0.12.0.dev20211119+cu102
```

## Dataset
The CQ500 dataset can be downloaded from [here](http://headctstudy.qure.ai/dataset). It should then be processed accordingly and divided with the following structure (in case you wanted to use the provided dataset class): 
```
dataset
|
|--- TRAIN
|    |--- head_volumes
|    |--- stripped_volumes
|
|--- VAL 
|    |--- head_volumes
|    |--- stripped_volumes
|
|--- TEST
     |
     |--- ATY
     |    |--- head_volumes
     |    |--- stripped_volumes
     |
     |--- FRAC      
     |    |--- head_volumes
     |    |--- stripped_volumes
     |        
    ...
```

## Running!
In order to run training, you should run the following command with the chosen parameters (here we chose toy parameters as example):
```
python src/train_lightning.py --gpu 0 --architecture AE --dataset_dir ./data --stripped False --output_dir ./runs
```
If you then would like to run inference/testing, you should memorize the path to your checkpoint and pass it to the following command:
```
python src/test_lightning.py --gpu 0 --architecture AE --checkpoint ./runs/AE-stripped/checkpoints/last.ckpt --stripped True --dataset_dir ./data
```

## Checking the results
All the testing data will be saved in folder `./<OUTPUT_DIR>/<ARCHITECTURE_NAME>/testing`. You can run Tensorboard to visualize losses, reconstructions, residuals, PR curves, confusion matrices and violin plots:
```
tensorboard --logdir ./runs/AE-stripped/testing/
```
Metrics are also saved inside this folder in Excel file format.
Good luck!
