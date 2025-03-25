# 📖 MAT (TCSVT)

### MAT: Multi-Range Attention Transformer for Efficient Image Super-Resolution [[Paper Link]](https://arxiv.org/pdf/2411.17214)

Chengxing Xie, Xiaoming Zhang, Linze Li, Yuqian Fu, Biao Gong, Tianrui Li and Kai Zhang

## Environment

**Recommend using tools similar to Miniconda for environment management.**

- Platforms: Ubuntu, CUDA >= 11.8
- [Python >= 3.10](https://www.python.org/) (Recommend using [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main))
- [PyTorch >= 2.1](https://pytorch.org/)
- [NATTEN >= 0.17.3](https://github.com/SHI-Labs/NATTEN/releases)
- [BasicSR == 1.4.2](https://github.com/XPixelGroup/BasicSR) **(Recommend using our implementation)**

The environment we use is: `Python == 3.10, Pytorch == 2.2, CUDA == 12.1, NATTEN == 0.17.3`

### Installation
```
# Clone the repo
git clone https://github.com/stella-von/MAT.git

# Install dependent packages
cd MAT
conda create -n mat python=3.10
conda activate mat

# Install Pytorch & NATTEN (for example)
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install natten==0.17.3+torch220cu121 -f https://shi-labs.com/natten/wheels

# Install BasicSR
pip install -r requirements.txt
python setup.py develop
```
You can also refer to [this page](https://github.com/XPixelGroup/BasicSR/blob/master/docs/INSTALL.md) for installation.

## Data Preparation

Please refer to [datasets/REDAME.md](datasets/README.md) for data preparation.

## How To Test

- Refer to `./options/test/MAT` for the configuration file of the model to be tested, and prepare the testing data and pretrained model.
- The pretrained models are available in [Google Drive](https://drive.google.com/drive/folders/1YSJSW_OjlxoefO6k6Ld1T0y1NOH82zzc?usp=sharing) | [Baidu Netdisk](https://pan.baidu.com/s/19PqiivK7XlawytvStGruFQ?pwd=x1mu).
- Then run the follwing codes (taking `MAT_light_x4.pth` as an example):

```
python basicsr/test.py -opt options/test/MAT/test_MAT_light_x4.yml
```

The testing results will be saved in the `./results` folder.

- Refer to `./inference/inference_mat.py` for **inference** without the ground truth image.
- Refer to `./basicsr/calculate_params_flops.py` for calculating the **parameters and flops.**

## How To Train

- Refer to `./options/train/MAT` for the configuration file of the model to train.
- Preparation of training data can refer to [this page](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md).
- The single GPU training command is as follows:

```
python basicsr/train.py -opt options/train/MAT/train_MAT_light_x2.yml
```

- The distributed training command is as follows:

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./scripts/dist_train.sh 8 options/train/MAT/train_MAT_light_x2.yml
```

More training commands can refer to [this page](https://github.com/XPixelGroup/BasicSR/blob/master/docs/TrainTest.md).

The training logs and weights will be saved in the `./experiments` folder.

## Pretrained Model & Visual Results
[Google Drive](https://drive.google.com/drive/folders/1YSJSW_OjlxoefO6k6Ld1T0y1NOH82zzc?usp=sharing) | [Baidu Netdisk](https://pan.baidu.com/s/19PqiivK7XlawytvStGruFQ?pwd=x1mu)

## Citation

If this work is helpful for your research, please consider citing the following BibTeX entry.

 ```
 @article{xie2025mat,
   author={Xie, Chengxing and Zhang, Xiaoming and Li, Linze and Fu, Yuqian and Gong, Biao and Li, Tianrui and Zhang, Kai},
   journal={IEEE Transactions on Circuits and Systems for Video Technology},
   title={MAT: Multi-Range Attention Transformer for Efficient Image Super-Resolution},
   year={2025},
   pages={1--13},
   doi={10.1109/TCSVT.2025.3553135}
 }
 ```

## Acknowledgement

This code is based on [BasicSR](https://github.com/XPixelGroup/BasicSR) and [NATTEN](https://github.com/SHI-Labs/NATTEN). Thanks for their awesome works.

## Contact

If you have any question, please email zxc0074869@gmail.com.