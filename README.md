# LEFormer: A Hybrid CNN-Transformer Architecture for Accurate Lake Extraction from Remote Sensing Imagery

[//]: # (![]&#40;resources/overall_architecture_diagram.jpg&#41;)
<p align="center">
    <img src="./resources/overall_architecture_diagram.jpg" height="550">
</p>

Figure 1: Overview architecture of LEFormer, consisting of four modules: (1) a hierarchical CNN encoder that extracts local features; (2) a  hierarchical Transformer encoder that captures global features; (3) a  cross-encoder fusion module that modulates local and global features from CNN and Transformer encoders; (4) a lightweight decoder that fuses the multi-scale features from the cross-encoder fusion module to predict the lake mask accurately.

The repository contains official PyTorch implementations of training and evaluation codes and pre-trained models for **LEFormer**.

[//]: # (The paper is in [Here]&#40;https://arxiv.org/pdf/2209.08575.pdf&#41;.)

The code is based on [MMSegmentaion v0.30.0](https://github.com/open-mmlab/MMSegmentation/tree/v0.30.0).

## Installation

For install and data preparation, please refer to the guidelines in [MMSegmentation v0.30.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.30.0).

An example (works for me): ```CUDA 10.1``` and  ```pytorch 1.6.0``` 

```
pip install -U openmim
mim install mmcv-full
cd LEFormer && pip install -e . --user
```

## Datasets Preparation

[//]: # (The Surface Water dataset &#40;SW dataset&#41; and Qinghai-Tibet Plateau Lake dataset &#40;QTPL dataset&#41; can be  download from [here]&#40;https://pan.baidu.com/s/1H2d6h3p3PtZw-g7PhNx9Tw?pwd=p0t7&#41;. )
After the paper has been accepted, we will make the download links for the Surface Water dataset (SW dataset) and the Qinghai-Tibet Plateau Lake dataset (QTPL dataset) that we used available.

The structure of datasets are aligned as follows:
```
SW or QTPL

├── annotations
│　　├── training 
│　　└── validation 
├── binary_annotations
│　　├── training 
│　　└── validation 
└── images  
 　　├── training 
　 　└── validation 
```
Alternatively, the datasets can be recreated to randomly split the datasets into training and testing sets, based on the original datasets.  

The original SW dataset is freely available for download [here](https://aistudio.baidu.com/aistudio/datasetdetail/75148).

The original QTPL dataset is freely available for download [here](http://www.ncdc.ac.cn/portal/metadata/b4d9fb27-ec93-433d-893a-2689379a3fc0).

Example: split ```Surface Water```:
```python
python tools/data_split.py --dataset_type sw --dataset_path /path/to/your/surface_water/train_data --save_path /path/to/save/dataset
```

Example: split ```Qinghai-Tibet Plateau Lake```:
```python
python tools/data_split.py --dataset_type qtpl --dataset_path /path/to/your/LakeWater --save_path /path/to/save/dataset
```


## Training

We use 1 GPU for training by default. Make sure you have modified the `data_root` variable in [sw_256x256.py](local_configs/_base_/datasets/sw_256x256.py) or [qtpl_256x256.py](local_configs/_base_/datasets/qtpl_256x256.py).    

Example: train ```LEFormer``` on ```Surface Water```:

```python
python tools/train.py local_configs/leformer/leformer_256x256_sw_160k.py
```

## Evaluation
To evaluate the model. Make sure you have modified the `data_root` variable in [sw_256x256.py](local_configs/_base_/datasets/sw_256x256.py) or [qtpl_256x256.py](local_configs/_base_/datasets/qtpl_256x256.py).  

Example: evaluate ```LEFormer``` on ```Surface Water```:

```python
python tools/test.py local_configs/leformer/leformer_256x256_sw_160k.py local_configs/pretrained_models/leformer_sw.pth --eval mIoU mFscore
```

## FLOPs

To calculate FLOPs for a model.

Example: calculate ```LEFormer``` on ```Surface Water```:

```python
python tools/get_flops.py local_configs/leformer/leformer_256x256_sw_160k.py --shape 256 256
```

## Acknowledgment

Our implementation is mainly based on [MMSegmentaion](https://github.com/open-mmlab/mmsegmentation/tree/v0.30.0), [Segformer](https://github.com/NVlabs/SegFormer) and [PoolFormer](https://github.com/sail-sg/poolformer). Thanks for their authors.


[//]: # (## LICENSE)

[//]: # ()
[//]: # (This repo is under the Apache-2.0 license. For commercial use, please contact the authors. )


## Supplement 
### Quantitative results of ablation study

<p align="center">
    <img src="./resources/ablation_study_1.jpg" height="600">
</p>

Figure 2: Visualization results of ablation studies on the number of Pooling Transformer Layers. **_L_** denotes the number of Pooling Transformer Layer.

<p align="center">
    <img src="./resources/ablation_study_2.jpg" height="550">
</p>

<p align="center">
    Figure 3: Visualization results of ablation studies on the CE, MSCA, TE and pooling operator modules.
</p>

