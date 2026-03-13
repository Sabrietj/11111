# 图神经网络工具DGL的安装步骤

[TOC]

## 一、问题

其中dgl的安装最为复杂。进入python环境以后import dgl，发现如下的错误。
```shell
$ python
Python 3.12.3 (tags/v3.12.3:f6650f9, Apr  9 2024, 14:05:25) [MSC v.1938 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import dgl
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "C:\Users\csqjxiao\.venv\Lib\site-packages\dgl\__init__.py", line 16, in <module>
    from . import (
  File "C:\Users\csqjxiao\.venv\Lib\site-packages\dgl\dataloading\__init__.py", line 13, in <module>
    from .dataloader import *
  File "C:\Users\csqjxiao\.venv\Lib\site-packages\dgl\dataloading\dataloader.py", line 27, in <module>
    from ..distributed import DistGraph
  File "C:\Users\csqjxiao\.venv\Lib\site-packages\dgl\distributed\__init__.py", line 5, in <module>
    from .dist_graph import DistGraph, DistGraphServer, edge_split, node_split
  File "C:\Users\csqjxiao\.venv\Lib\site-packages\dgl\distributed\dist_graph.py", line 11, in <module>
    from .. import backend as F, graphbolt as gb, heterograph_index
  File "C:\Users\csqjxiao\.venv\Lib\site-packages\dgl\graphbolt\__init__.py", line 36, in <module>
    load_graphbolt()
  File "C:\Users\csqjxiao\.venv\Lib\site-packages\dgl\graphbolt\__init__.py", line 26, in load_graphbolt
    raise FileNotFoundError(
FileNotFoundError: Cannot find DGL C++ graphbolt library at C:\Users\csqjxiao\.venv\Lib\site-packages\dgl\graphbolt\graphbolt_pytorch_2.4.1.dll
```
主要原因是自从2024.06.27，dgl就停止提供Windows和MacOS操作系统上的prebuild软件安装包。具体可以查看https://data.dgl.ai/wheels/repo.html，最新支持的dgl版本只有2.2.1。所以基本不能用pip和conda直接安装。

## 二、安装步骤

### 在Ubuntu系统下
可以[按照官方步骤从Prebuilt安装DGL](https://dgl.ac.cn/pages/start.html)。在 Ubuntu 系统下安装 DGL 前，建议先确认当前环境中的 PyTorch 版本和 CUDA 版本，再选择对应的安装命令：

1. 首先检查已安装的 PyTorch 版本和 CUDA 版本：
   ```bash
   # 查看PyTorch版本
   pip list | grep torch
   
   # 查看CUDA版本（如果已安装）
   nvcc -V
   # 或通过PyTorch查看
   python -c "import torch; print(torch.version.cuda)"
   ```

2. Pytorch的版本要和cuda版本，以及dgl需要的版本搭配。比如，[按照官方步骤从Prebuilt安装DGL](https://dgl.ac.cn/pages/start.html)要求Pytorch 2.4。在[Pytorch的官方网站](https://pytorch.org/get-started/previous-versions/)上，有CUDA 10.1的匹配命令。
  ```shell
  # CUDA 11.8
  pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0 -f https://download.pytorch.org/whl/torch_stable.html
  ```

3. 根据查询结果，选择对应的安装命令：若 PyTorch 版本为 2.4 且 CUDA 版本为 12.4：
   ```shell
   pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
   ```
   由于学校的大数据服务中心无法翻墙，使用pip安装时会出现连接失败的问题。我们可以直接找一个能够翻墙的主机，将如`dgl-2.4.0+cu124-cp312-cp312-manylinux1_x86_64.whl`的whl文件下载到本地，然后上传到大数据中心。再`pip install <文件名>.whl`即可完成dgl安装。
   
   注：如果页面链接转义失败，可以直接到html页面（即`https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html`），复制对应版本`.whl`文件下载链接安装:`pip install "链接"`
4.  如果你的 PyTorch 版本或 CUDA 版本与上述不符，可以到 [DGL 官方文档的安装页面](https://dgl.ac.cn/pages/start.html)查找对应版本的安装链接。注意：如果使用 CPU 版本的 PyTorch，应选择对应的 CPU 版本 DGL 安装包。

### 在Windows系统下

可以从下面的百度网盘链接下载完整的dgl的源代码文件夹。
```
通过网盘分享的文件：dgl.zip
链接: https://pan.baidu.com/s/14fxlt3EfX-aQP2R8k2ZV4g?pwd=j7rf 
提取码: j7rf 
```

可以[按照官方步骤从源代码安装DGL](https://docs.dgl.ai/install/index.html#install-from-source)。我在Windows下的安装步骤：
* Download the source files from GitHub.
```shell
git config --global http.proxy "http://127.0.0.1:7897"
git config --global https.proxy "http://127.0.0.1:7897"

set http_proxy="http://127.0.0.1:7897"
set https_proxy="http://127.0.0.1:7897"

git clone --recurse-submodules https://github.com/dmlc/dgl.git
cd dgl
```
* You can build DGL with MSBuild. With [MS Build Tools](https://go.microsoft.com/fwlink/?linkid=840931) and [CMake](https://cmake.org/download/) on Windows installed, run the following in VS2022 x64 Native tools command prompt. 注意：记着将`C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin`的路径加入系统环境变量Path。这样才能找到msbuild。
* CPU only build:
```shell
RM build
MD build
CD build
cmake -DCMAKE_CXX_FLAGS="/DDGL_EXPORTS" -DCMAKE_CONFIGURATION_TYPES="Release" -DDMLC_FORCE_SHARED_CRT=ON .. -G "Visual Studio 17 2022"
msbuild dgl.sln /m
CD ..\python
python setup.py install
```
* CUDA build:
```shell
RM build
MD build
CD build
cmake -DCMAKE_CXX_FLAGS="/DDGL_EXPORTS" -DCMAKE_CONFIGURATION_TYPES="Release" -DDMLC_FORCE_SHARED_CRT=ON -DUSE_CUDA=ON .. -G "Visual Studio 17 2022"
msbuild dgl.sln /m
CD ..\python
python setup.py install
```
重新验证dgl的安装。
```shell
$ python
Python 3.12.3 (tags/v3.12.3:f6650f9, Apr  9 2024, 14:05:25) [MSC v.1938 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import dgl
>>>
```

### 在MacOS系统下

可以[按照官方步骤从源代码安装DGL](https://docs.dgl.ai/install/index.html#install-from-source)。我在Windows下的安装步骤：
* Download the source files from GitHub.
```shell
git config --global http.proxy "http://127.0.0.1:7897"
git config --global https.proxy "http://127.0.0.1:7897"

export http_proxy="http://127.0.0.1:7897"
export https_proxy="http://127.0.0.1:7897"

git clone --recurse-submodules https://github.com/dmlc/dgl.git
cd dgl
```
Build and Install DGL.
```shell
mkdir build
cd build
cmake -DUSE_OPENMP=off -DUSE_LIBXSMM=OFF ..
make -j4
cd ../python
python setup.py install
python setup.py build_ext --inplace
```