# LEFormer: A Hybrid CNN-Transformer Architecture for Accurate Lake Extraction from Remote Sensing Imagery

![](resources/overall_architecture_diagram.jpg)

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

### Datasets Preparation
The Surface Water dataset (SW dataset) can be  download from [here](https://aistudio.baidu.com/aistudio/datasetdetail/75148). 

The Qinghai-Tibet Plateau Lake dataset (QTPL dataset) can be  download from [here](http://www.ncdc.ac.cn/portal/metadata/b4d9fb27-ec93-433d-893a-2689379a3fc0). 

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
python tools/data_split.py --dataset_type sw --dataset_path /path/to/your/dataset --save_path /path/to/save/dataset
```


## Training

We use 1 GPU for training by default.  

Example: train ```LEFormer``` on ```Surface Water```:

```python
python tools/train.py local_configs/leformer/leformer_256x256_SW_160k.py
```

## Evaluation
To evaluate the model. 

Example: evaluate ```LEFormer``` on ```Surface Water```:

```python
python tools/test.py local_configs/leformer/leformer_256x256_SW_160k.py local_configs/pretrained_models/leformer_sw.pth --eval mIoU mFscore
```

## FLOPs

To calculate FLOPs for a model.

Example: calculate ```LEFormer``` on ```Surface Water```:

```python
python tools/get_flops.py local_configs/leformer/leformer_256x256_SW_160k.py --shape 256 256
```

## Acknowledgment

Our implementation is mainly based on [MMSegmentaion](https://github.com/open-mmlab/mmsegmentation/tree/v0.30.0), [Segformer](https://github.com/NVlabs/SegFormer) and [PoolFormer](https://github.com/sail-sg/poolformer). Thanks for their authors.


[//]: # (## LICENSE)

[//]: # ()
[//]: # (This repo is under the Apache-2.0 license. For commercial use, please contact the authors. )


## Supplement 
### Quantitative results of ablation study

[//]: # (![]&#40;resources/ablation_study_1.jpg&#41;)
<div align="center">
    <img src="./resources/ablation_study_1.jpg" height="450">
    <img src="./resources/ablation_study_2.jpg" height="450">
</div>
<p align="center">
  <p align="center">
  Figure 1: Visualization results of ablation studies on the SW and QTPL datasets. **Left**: Ablation studies on the number of Pooling Transformer Layers. *L* denotes the number of layers; **Right**: Ablation studies on the CE, MSCA, TE, and pooling operator modules.
</p>

[//]: # ([//]: # &#40;![]&#40;resources/ablation_study_3.jpg&#41;&#41;)
[//]: # (<!-- ![image]&#40;resources/ablation_study_3.jpg&#41; -->)

[//]: # (<div align="center">)

[//]: # (  <img src="./resources/ablation_study_3.jpg" height="500">)

[//]: # (</div>)

[//]: # (<p align="center">)

[//]: # (  Figure 2: Visualization results of our proposed LEFormer and other methods on the SW and QTPL datasets for the lake mask extraction. The white circles indicate apparent differences.)

[//]: # (</p>)
