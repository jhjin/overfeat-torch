## OverFeat-Torch7 Wrapper

OverFeat is a Convolutional Network-based image classifier and feature extractor from NYU.
The original can be found in the repository: https://github.com/sermanet/OverFeat
This application loads weights from OverFeat and construct a network for vanilla Torch7.
Torch7 and extra packages (image, nn, torchffi) should be properly installed.


## Install

Run the shell script to download weights and build this library.

```bash
sh install.sh
```


## Run demo

Run the command below.
By default, the script loads a small network,
and categorizes the `bee.jpg` image using `nn` backend (on CPU).

```bash
th run.lua
```

For example, if you want to run a big model on GPU using `cudnn` library in a memory-efficient manner (inplace opertor), use the command instead.

```bash
th run.lua --network big --backend cudnn --inplace
```
