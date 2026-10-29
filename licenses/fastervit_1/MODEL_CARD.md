---
license: other
datasets:
- imagenet-1k
---
[**FasterViT: Fast Vision Transformers with Hierarchical Attention**](https://arxiv.org/abs/2306.06189).


FasterViT achieves a new SOTA Pareto-front in
terms of accuracy vs. image throughput without extra training data !

<p align="center">
<img src="https://github.com/NVlabs/FasterViT/assets/26806394/253d1a2e-b5f5-4a9b-a362-6cdd16bfccc1" width=62% height=62% 
class="center">
</p>

Note: Please use the [**latest NVIDIA TensorRT release**](https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/index.html) to enjoy the benefits of optimized FasterViT ops. 


## Quick Start

We can import pre-trained FasterViT models with **1 line of code**. First, FasterViT can be simply installed by:

```bash
pip install fastervit
```

A pretrained FasterViT model with default hyper-parameters can be created as in the following:

```python
>>> from fastervit import create_model

# Define fastervit-0 model with 224 x 224 resolution

>>> model = create_model('faster_vit_0_224', 
                          pretrained=True,
                          model_path="/tmp/faster_vit_0.pth.tar")
```

`model_path` is used to set the directory to download the model.

We can also simply test the model by passing a dummy input image. The output is the logits:

```python
>>> import torch

>>> image = torch.rand(1, 3, 224, 224)
>>> output = model(image) # torch.Size([1, 1000])
```

We can also use the any-resolution FasterViT model to accommodate arbitrary image resolutions. In the following, we define an any-resolution FasterViT-0
model with input resolution of 576 x 960, window sizes of 12 and 6 in 3rd and 4th stages, carrier token size of 2 and embedding dimension of
64:

```python
>>> from fastervit import create_model

# Define any-resolution FasterViT-0 model with 576 x 960 resolution
>>> model = create_model('faster_vit_0_any_res', 
                          resolution=[576, 960],
                          window_size=[7, 7, 12, 6],
                          ct_size=2,
                          dim=64,
                          pretrained=True)
```
Note that the above model is intiliazed from the original ImageNet pre-trained FasterViT with original resolution of 224 x 224. As a result, missing keys and mis-matches could be expected since we are addign new layers (e.g. addition of new carrier tokens, etc.) 

We can simply test the model by passing a dummy input image. The output is the logits:

```python
>>> import torch

>>> image = torch.rand(1, 3, 576, 960)
>>> output = model(image) # torch.Size([1, 1000])
```

--- 

## Results + Pretrained Models

### ImageNet-1K
**FasterViT ImageNet-1K Pretrained Models**

<table>
  <tr>
    <th>Name</th>
    <th>Acc@1(%)</th>
    <th>Acc@5(%)</th>
    <th>Throughput(Img/Sec)</th>
    <th>Resolution</th>
    <th>#Params(M)</th>
    <th>FLOPs(G)</th>
    <th>Download</th>
  </tr>

<tr>
    <td>FasterViT-0</td>
    <td>82.1</td>
    <td>95.9</td>
    <td>5802</td>
    <td>224x224</td>
    <td>31.4</td>
    <td>3.3</td>
    <td><a href="https://drive.google.com/uc?export=download&id=1twI2LFJs391Yrj8MR4Ui9PfrvWqjE1iB">model</a></td>
</tr>

<tr>
    <td>FasterViT-1</td>
    <td>83.2</td>
    <td>96.5</td>
    <td>4188</td>
    <td>224x224</td>
    <td>53.4</td>
    <td>5.3</td>
    <td><a href="https://drive.google.com/uc?export=download&id=1r7W10n5-bFtM3sz4bmaLrowN2gYPkLGT">model</a></td>
</tr>

<tr>
    <td>FasterViT-2</td>
    <td>84.2</td>
    <td>96.8</td>
    <td>3161</td>
    <td>224x224</td>
    <td>75.9</td>
    <td>8.7</td>
    <td><a href="https://drive.google.com/uc?export=download&id=1n_a6s0pgi0jVZOGmDei2vXHU5E6RH5wU">model</a></td>
</tr>

<tr>
    <td>FasterViT-3</td>
    <td>84.9</td>
    <td>97.2</td>
    <td>1780</td>
    <td>224x224</td>
    <td>159.5</td>
    <td>18.2</td>
    <td><a href="https://drive.google.com/uc?export=download&id=1tvWElZ91Sia2SsXYXFMNYQwfipCxtI7X">model</a></td>
</tr>

<tr>
    <td>FasterViT-4</td>
    <td>85.4</td>
    <td>97.3</td>
    <td>849</td>
    <td>224x224</td>
    <td>424.6</td>
    <td>36.6</td>
    <td><a href="https://drive.google.com/uc?export=download&id=1gYhXA32Q-_9C5DXel17avV_ZLoaHwdgz">model</a></td>
</tr>

<tr>
    <td>FasterViT-5</td>
    <td>85.6</td>
    <td>97.4</td>
    <td>449</td>
    <td>224x224</td>
    <td>975.5</td>
    <td>113.0</td>
    <td><a href="https://drive.google.com/uc?export=download&id=1mqpai7XiHLr_n1tjxjzT8q369xTCq_z-">model</a></td>
</tr>

<tr>
    <td>FasterViT-6</td>
    <td>85.8</td>
    <td>97.4</td>
    <td>352</td>
    <td>224x224</td>
    <td>1360.0</td>
    <td>142.0</td>
    <td><a href="https://drive.google.com/uc?export=download&id=12jtavR2QxmMzcKwPzWe7kw-oy34IYi59">model</a></td>
</tr>

</table>

### ImageNet-21K
**FasterViT ImageNet-21K Pretrained Models (ImageNet-1K Fine-tuned)**

<table>
  <tr>
    <th>Name</th>
    <th>Acc@1(%)</th>
    <th>Acc@5(%)</th>
    <th>Resolution</th>
    <th>#Params(M)</th>
    <th>FLOPs(G)</th>
    <th>Download</th>
  </tr>

<tr>
    <td>FasterViT-4-21K-224</td>
    <td>86.6</td>
    <td>97.8</td>
    <td>224x224</td>
    <td>271.9</td>
    <td>40.8</td>
    <td><a href="https://huggingface.co/ahatamiz/FasterViT/resolve/main/fastervit_4_21k_224_w14.pth.tar">model</a></td>
</tr>

<tr>
    <td>FasterViT-4-21K-384</td>
    <td>87.6</td>
    <td>98.3</td>
    <td>384x384</td>
    <td>271.9</td>
    <td>120.1</td>
    <td><a href="https://huggingface.co/ahatamiz/FasterViT/resolve/main/fastervit_4_21k_384_w24.pth.tar">model</a></td>
</tr>

<tr>
    <td>FasterViT-4-21K-512</td>
    <td>87.8</td>
    <td>98.4</td>
    <td>512x512</td>
    <td>271.9</td>
    <td>213.5</td>
    <td><a href="https://huggingface.co/ahatamiz/FasterViT/resolve/main/fastervit_4_21k_512_w32.pth.tar">model</a></td>
</tr>

<tr>
    <td>FasterViT-4-21K-768</td>
    <td>87.9</td>
    <td>98.5</td>
    <td>768x768</td>
    <td>271.9</td>
    <td>480.4</td>
    <td><a href="https://huggingface.co/ahatamiz/FasterViT/resolve/main/fastervit_4_21k_768_w48.pth.tar">model</a></td>
</tr>

</table>

### Robustness (ImageNet-A - ImageNet-R - ImageNet-V2)

All models use `crop_pct=0.875`. Results are obtained by running inference on ImageNet-1K pretrained models without finetuning.
<table>
  <tr>
    <th>Name</th>
    <th>A-Acc@1(%)</th>
    <th>A-Acc@5(%)</th>
    <th>R-Acc@1(%)</th>
    <th>R-Acc@5(%)</th>
    <th>V2-Acc@1(%)</th>
    <th>V2-Acc@5(%)</th>
  </tr>

<tr>
    <td>FasterViT-0</td>
    <td>23.9</td>
    <td>57.6</td>
    <td>45.9</td>
    <td>60.4</td>
    <td>70.9</td>
    <td>90.0</td>
</tr>

<tr>
    <td>FasterViT-1</td>
    <td>31.2</td>
    <td>63.3</td>
    <td>47.5</td>
    <td>61.9</td>
    <td>72.6</td>
    <td>91.0</td>
</tr>

<tr>
    <td>FasterViT-2</td>
    <td>38.2</td>
    <td>68.9</td>
    <td>49.6</td>
    <td>63.4</td>
    <td>73.7</td>
    <td>91.6</td>
</tr>

<tr>
    <td>FasterViT-3</td>
    <td>44.2</td>
    <td>73.0</td>
    <td>51.9</td>
    <td>65.6</td>
    <td>75.0</td>
    <td>92.2</td>
</tr>

<tr>
    <td>FasterViT-4</td>
    <td>49.0</td>
    <td>75.4</td>
    <td>56.0</td>
    <td>69.6</td>
    <td>75.7</td>
    <td>92.7</td>
</tr>

<tr>
    <td>FasterViT-5</td>
    <td>52.7</td>
    <td>77.6</td>
    <td>56.9</td>
    <td>70.0</td>
    <td>76.0</td>
    <td>93.0</td>
</tr>

<tr>
    <td>FasterViT-6</td>
    <td>53.7</td>
    <td>78.4</td>
    <td>57.1</td>
    <td>70.1</td>
    <td>76.1</td>
    <td>93.0</td>
</tr>

</table>

A, R and V2 denote ImageNet-A, ImageNet-R and ImageNet-V2 respectively. 

## Citation

Please consider citing FasterViT if this repository is useful for your work. 

```
@article{hatamizadeh2023fastervit,
  title={FasterViT: Fast Vision Transformers with Hierarchical Attention},
  author={Hatamizadeh, Ali and Heinrich, Greg and Yin, Hongxu and Tao, Andrew and Alvarez, Jose M and Kautz, Jan and Molchanov, Pavlo},
  journal={arXiv preprint arXiv:2306.06189},
  year={2023}
}
```


## Licenses

Copyright © 2023, NVIDIA Corporation. All rights reserved.

This work is made available under the NVIDIA Source Code License-NC. Click [here](https://huggingface.co/nvidia/FasterViT/blob/main/LICENSE) to view a copy of this license.

For license information regarding the timm repository, please refer to its [repository](https://github.com/rwightman/pytorch-image-models).

For license information regarding the ImageNet dataset, please see the [ImageNet official website](https://www.image-net.org/). 

## Acknowledgement
This repository is built on top of the [timm](https://github.com/huggingface/pytorch-image-models) repository. We thank [Ross Wrightman](https://rwightman.com/) for creating and maintaining this high-quality library.