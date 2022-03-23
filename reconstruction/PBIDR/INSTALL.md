## MANUAL INSTALL

```bash
conda create -n pbidr python=3.7
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html
conda install -c conda-forge scikit-image shapely rtree pyembree
pip install trimesh[all]
```

#### Install Pytorch3D

```bash
wget https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
tar xzf 1.10.0.tar.gz
export CUB_HOME=$PWD/cub-1.10.0
wget https://github.com/facebookresearch/pytorch3d/archive/refs/tags/v0.4.0.tar.gz
tar xzf v0.4.0.tar.gz
export TORCH_CUDA_ARCH_LIST="7.5"
cd pytorch3d-0.4.0 && pip install -e .
```

#### Install Other Requirments

```bash
pip install requirements.txt
```

