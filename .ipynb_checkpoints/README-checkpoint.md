# TMa_UNet

**TMa_UNet : Tri-Directional Mamba U-Net with GSC2**

![](image/TMa_UNet.png)


## Environment install
May be you should have an environment follow :

```bash
python 3.10 + torch 2.0.1 + torchvision 0.15.2 (cuda 11.8)
```

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
Put data in data/dataset/ 

For example: **data/dataset/AffinedManualSegImageNIfT** 、 **data/dataset/RawImageNIfT**

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

Run 1、2 for the test set

Modify function parameters: You need to adjust the parameters of the function **get_train_val_test_loader_from_train**

The function takes several arguments:

    data_dir: The directory where the data is stored.
    train_number: The number of samples to be used for training.
    val_number: The number of samples to be used for validation.
    test_number: The number

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
