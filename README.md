May be you should have an environment follow :python 3.10 + torch 2.0.1 + torchvision 0.15.2 (cuda 11.8)

## Environment install

### Install causal-conv1d

```bash
cd causal-conv1d

pip install -e .
```

### Install mamba

```bash
cd mamba

pip install -e .
```

### Install monai 

```bash
pip install monai
```

## Preprocessing, training, testing, inference, and metrics computation

### Preprocessing
Put data in **data/dataset/AffinedManualSegImageNIfT** 、 **data/dataset/RawImageNIfT**

```bash 
python 1_rename.py
```

```bash
python 2_preprocess.py
```

### Training 

```bash 
python 3_train.py
```

The training logs and checkpoints are saved in: **logdir = f"./logs/TMa-UNet"**

### Inference 

Before inference, run 1、2 for the test set and modify function parameters **get_train_val_test_loader_from_train(data_dir, train_number = 0, val_number = 0,test_number=len(test_set))**

```bash 
python 4_predict.py
```

### Evaluation

Put GT in  **prediction_results/GT**

run **prediction_results/Evalu.ipynb**


## Acknowledgement
Many thanks for these repos for their great contribution!

[https://github.com/ge-xing/SegMamba](https://github.com/ge-xing/SegMamba)

[https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)

[https://github.com/Project-MONAI/MONAI](https://github.com/Project-MONAI/MONAI)

[https://github.com/hustvl/Vim](https://github.com/hustvl/Vim)

[https://github.com/bowang-lab/U-Mamba](https://github.com/bowang-lab/U-Mamba)
