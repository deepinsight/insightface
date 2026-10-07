# Distributed ArcFace Training in PyTorch

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/killing-two-birds-with-one-stone-efficient/face-verification-on-ijb-c)](https://paperswithcode.com/sota/face-verification-on-ijb-c?p=killing-two-birds-with-one-stone-efficient)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/killing-two-birds-with-one-stone-efficient/face-verification-on-ijb-b)](https://paperswithcode.com/sota/face-verification-on-ijb-b?p=killing-two-birds-with-one-stone-efficient)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/killing-two-birds-with-one-stone-efficient/face-verification-on-agedb-30)](https://paperswithcode.com/sota/face-verification-on-agedb-30?p=killing-two-birds-with-one-stone-efficient)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/killing-two-birds-with-one-stone-efficient/face-verification-on-cfp-fp)](https://paperswithcode.com/sota/face-verification-on-cfp-fp?p=killing-two-birds-with-one-stone-efficient)

This is the official PyTorch implementation of **ArcFace** (Additive Angular Margin Loss for Deep Face Recognition). This repository provides an efficient distributed training framework with comprehensive support for large-scale face recognition datasets.

## Table of Contents

- [What is ArcFace?](#what-is-arcface)
- [Key Features](#key-features)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Quick Start](#quick-start)
- [Training Guide](#training-guide)
- [Dataset Preparation](#dataset-preparation)
- [Configuration Files](#configuration-files)
- [Model Zoo](#model-zoo)
- [Performance Benchmarks](#performance-benchmarks)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Citation](#citation)

## What is ArcFace?

ArcFace is a state-of-the-art face recognition method that adds an angular margin penalty to the softmax loss. This makes the learned face features more discriminative, leading to better recognition accuracy. This implementation provides:

- **High Performance**: Optimized for fast training on large datasets
- **Scalability**: Can train on datasets with millions of identities
- **Flexibility**: Supports various backbones including CNNs and Vision Transformers
- **Production Ready**: Includes ONNX export for easy deployment

## Key Features

### Distributed Training
- **Multi-GPU Training**: Efficiently use all GPUs on a single machine
- **Multi-Node Training**: Scale across multiple machines for very large datasets
- **Automatic Mixed Precision (AMP)**: Faster training with lower memory usage
- **Gradient Checkpointing**: Train larger models with limited GPU memory

### Advanced Training Techniques
- **Partial FC**: Efficiently train on datasets with millions of identities (up to 29M)
- **Memory Optimization**: Sparse softmax sampling reduces memory requirements
- **Fast Data Loading**: Optional NVIDIA DALI support for accelerated data loading
- **Multiple Backbones**: ResNet, MobileFaceNet, and Vision Transformers (ViT-T/S/B/L)

### Large-Scale Dataset Support
- WebFace42M (2M identities, 42.5M images)
- Glint360K (360K identities, 17.1M images)
- MS1MV3 (93K identities, 5.2M images)
- Custom datasets

## System Requirements

### Minimum Requirements
- **Operating System**: Linux (Ubuntu 18.04+ recommended)
- **Python**: 3.7 or higher
- **PyTorch**: 1.12.0 or higher
- **CUDA**: 10.2 or higher
- **GPU**: NVIDIA GPU with at least 12GB VRAM (e.g., Tesla V100, RTX 3090)
- **RAM**: 32GB system RAM minimum
- **Storage**: At least 100GB free space for datasets

### Recommended Setup
- **GPU**: 8x NVIDIA Tesla V100 (32GB) or A100 (40GB/80GB)
- **CPU**: 32+ cores
- **RAM**: 128GB or more
- **Storage**: NVMe SSD with 1TB+ space
- **Network**: 10Gbps+ for multi-node training

## Installation Guide

### Step 1: Clone the Repository

First, clone the InsightFace repository to your local machine:

```bash
# Clone the repository
git clone https://github.com/deepinsight/insightface.git

# Navigate to the ArcFace training directory
cd insightface/recognition/arcface_torch
```

### Step 2: Create a Virtual Environment (Recommended)

It's best practice to use a virtual environment to avoid package conflicts:

```bash
# Create a new virtual environment
python3 -m venv arcface_env

# Activate the virtual environment
# On Linux/Mac:
source arcface_env/bin/activate
# On Windows:
# arcface_env\Scripts\activate
```

### Step 3: Install PyTorch

Install PyTorch with CUDA support. Visit [pytorch.org](https://pytorch.org/get-started/locally/) to get the installation command for your system. Example for CUDA 11.8:

```bash
# Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Install Required Packages

Install all dependencies listed in the requirements file:

```bash
# Install required packages
pip install -r requirement.txt
```

The main dependencies include:
- `torch` - PyTorch deep learning framework
- `torchvision` - Computer vision utilities
- `tensorboard` - Training visualization
- `opencv-python` - Image processing
- `scikit-learn` - Machine learning utilities
- `scikit-image` - Image processing
- `easydict` - Easy dictionary access
- `mxnet` - For reading MXNet record files

### Step 5: Install NVIDIA DALI (Optional but Recommended)

DALI significantly speeds up data loading. Install it with:

```bash
# Install NVIDIA DALI for CUDA 11.0
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110
```

For detailed DALI installation instructions, see [docs/install_dali.md](docs/install_dali.md).

### Step 6: Verify Installation

Verify that everything is installed correctly:

```bash
# Check PyTorch installation
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# Check if GPU is accessible
python -c "import torch; print('Number of GPUs:', torch.cuda.device_count()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

Expected output should show:
- PyTorch version 1.12.0 or higher
- CUDA available: True
- Number of GPUs: 1 or more
- GPU Name: Your GPU model

## Quick Start

### Download a Small Dataset

For quick testing, download the MS1MV3 dataset:

```bash
# Create data directory
mkdir -p data

# Download MS1MV3 dataset (you'll need to get the download link)
# See the Datasets section below for download instructions
```

### Run Your First Training

Test single GPU training with a small config:

```bash
# Single GPU training (for testing)
python train_v2.py configs/ms1mv3_r50_onegpu
```

This will:
1. Load the MS1MV3 dataset
2. Initialize a ResNet-50 backbone
3. Start training with default hyperparameters
4. Save checkpoints to the `work_dirs` directory
5. Log training progress to TensorBoard

### Monitor Training Progress

Open another terminal and start TensorBoard:

```bash
# Start TensorBoard
tensorboard --logdir work_dirs

# Open your browser and go to http://localhost:6006
```

## Training Guide

### Understanding the Training Command

The basic training command structure is:

```bash
torchrun [distributed_options] train_v2.py [config_file]
```

- `torchrun`: PyTorch's distributed training launcher
- `distributed_options`: Options for distributed training (number of GPUs, nodes, etc.)
- `train_v2.py`: The training script
- `config_file`: Configuration file specifying model, dataset, and training parameters

### Single GPU Training

**When to use:** Testing, debugging, or when you only have one GPU.

```bash
python train_v2.py configs/ms1mv3_r50_onegpu
```

**Note:** Single GPU training is significantly slower and is only recommended for:
- Testing your setup
- Debugging code
- Small-scale experiments

For production training, always use multiple GPUs.

### Multi-GPU Training (Single Machine)

**When to use:** You have multiple GPUs on one machine (most common scenario).

```bash
# Use all available GPUs (e.g., 8 GPUs)
torchrun --nproc_per_node=8 train_v2.py configs/ms1mv3_r50

# Use specific number of GPUs (e.g., 4 GPUs)
torchrun --nproc_per_node=4 train_v2.py configs/ms1mv3_r50
```

**Parameters explained:**
- `--nproc_per_node=8`: Number of processes (GPUs) to use on this machine

**Example output:**
```
Rank 0: GPU 0 - Training started
Rank 1: GPU 1 - Training started
...
Epoch 1, Step 100, Loss: 0.45, LR: 0.1
```

### Multi-Node Training (Multiple Machines)

**When to use:** Training on very large datasets (WebFace42M) or when you need more compute power.

**Requirements:**
- Multiple machines with GPUs
- Machines can communicate over network
- Same dataset accessible on all machines (via NFS or local copy)

**Setup:**

On **Node 0** (master node):
```bash
torchrun \
  --nproc_per_node=8 \
  --nnodes=2 \
  --node_rank=0 \
  --master_addr="192.168.1.100" \
  --master_port=12581 \
  train_v2.py configs/wf42m_pfc02_16gpus_r100
```

On **Node 1** (worker node):
```bash
torchrun \
  --nproc_per_node=8 \
  --nnodes=2 \
  --node_rank=1 \
  --master_addr="192.168.1.100" \
  --master_port=12581 \
  train_v2.py configs/wf42m_pfc02_16gpus_r100
```

**Parameters explained:**
- `--nproc_per_node=8`: Number of GPUs per node
- `--nnodes=2`: Total number of nodes (machines)
- `--node_rank=0`: Rank of this node (0 for master, 1, 2, ... for workers)
- `--master_addr="192.168.1.100"`: IP address of the master node
- `--master_port=12581`: Port for communication (use any free port)

**Important notes:**
1. All nodes must use the same configuration file
2. The master node must be reachable from all worker nodes
3. All nodes should start training within a few seconds of each other
4. Use the master node's IP address (check with `ifconfig` or `ip addr`)

### Training with Vision Transformers

Vision Transformers (ViT) often achieve better accuracy than CNNs:

```bash
# Train ViT-B (Base) model
torchrun --nproc_per_node=8 train_v2.py configs/wf42m_pfc03_40epoch_8gpu_vit_b
```

**Available ViT models:**
- `ViT-T`: Tiny model (fastest, lower accuracy)
- `ViT-S`: Small model (balanced)
- `ViT-B`: Base model (recommended for most use cases)
- `ViT-L`: Large model (best accuracy, requires more memory)

### Resuming Training from Checkpoint

If training is interrupted, you can resume from the last checkpoint:

```bash
# Training will automatically resume from the latest checkpoint in work_dirs
python train_v2.py configs/ms1mv3_r50 --resume
```

### Using Different Backbones

The repository supports multiple backbone architectures:

**CNN Backbones:**
- `r50`: ResNet-50 (standard choice)
- `r100`: ResNet-100 (better accuracy, more compute)
- `r200`: ResNet-200 (best accuracy, highest compute)
- `mobilefacenet`: MobileFaceNet (fast, low memory)

**Transformer Backbones:**
- `vit_t`: Vision Transformer Tiny
- `vit_s`: Vision Transformer Small
- `vit_b`: Vision Transformer Base
- `vit_l`: Vision Transformer Large

## Dataset Preparation

### Available Datasets

| Dataset | Identities | Images | Size | Use Case |
|---------|-----------|---------|------|----------|
| MS1MV2 | 87K | 5.8M | ~50GB | Good starting point |
| MS1MV3 | 93K | 5.2M | ~45GB | Recommended for learning |
| Glint360K | 360K | 17.1M | ~150GB | Large-scale training |
| WebFace42M | 2M | 42.5M | ~400GB | State-of-the-art results |

### Downloading Datasets

#### MS1MV3 (Recommended for Beginners)

1. Download from the links in [recognition/_datasets_](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_#ms1m-retinaface)
2. Extract to `data/ms1m-retinaface-t1/`
3. The directory should contain:
   - `train.rec` - Training images in MXNet record format
   - `train.idx` - Index file for the record
   - `property` - Dataset properties

#### WebFace42M (For Advanced Users)

1. Follow the guide in [docs/prepare_webface42m.md](docs/prepare_webface42m.md)
2. This dataset requires significant storage (400GB+)
3. Preparation may take several hours

### Dataset Directory Structure

After downloading, your data directory should look like:

```
insightface/recognition/arcface_torch/
├── data/
│   ├── ms1m-retinaface-t1/
│   │   ├── train.rec
│   │   ├── train.idx
│   │   └── property
│   ├── glint360k/
│   │   ├── train.rec
│   │   ├── train.idx
│   │   └── property
│   └── webface42m/
│       ├── train.rec
│       ├── train.idx
│       └── property
```

### Preparing Custom Datasets

To train on your own face dataset:

1. **Organize your data:**
   ```
   my_dataset/
   ├── person1/
   │   ├── img1.jpg
   │   ├── img2.jpg
   │   └── ...
   ├── person2/
   │   ├── img1.jpg
   │   └── ...
   └── ...
   ```

2. **Convert to MXNet record format:**
   ```bash
   # See docs/prepare_custom_dataset.md for detailed instructions
   python tools/convert_to_rec.py --input my_dataset --output data/my_dataset
   ```

3. **Create a configuration file** for your dataset (copy and modify an existing config)

### Data Preprocessing for DALI

If you're using NVIDIA DALI for faster data loading, shuffle the rec files first:

```bash
# Shuffle the dataset for better DALI performance
python scripts/shuffle_rec.py data/ms1m-retinaface-t1

# This creates: data/shuffled_ms1m-retinaface-t1/train.rec
```

**Why shuffle?** DALI works more efficiently with shuffled data, leading to faster training.

## Configuration Files

Configuration files are located in the `configs/` directory. Each config file specifies:

### Configuration File Structure

```python
# configs/ms1mv3_r50.py
config = dict(
    # Network architecture
    network = "r50",           # Backbone: r50, r100, vit_b, etc.
    
    # Dataset settings
    dataset = "ms1mv3",        # Dataset name
    rec = "data/ms1m-retinaface-t1",  # Path to dataset
    num_classes = 93431,       # Number of identities in dataset
    num_image = 5179510,       # Total number of images
    
    # Training hyperparameters
    batch_size = 128,          # Batch size per GPU
    num_epoch = 20,            # Number of epochs
    warmup_epoch = 0,          # Learning rate warmup epochs
    
    # Learning rate settings
    lr = 0.1,                  # Initial learning rate
    momentum = 0.9,            # SGD momentum
    weight_decay = 5e-4,       # Weight decay (L2 regularization)
    
    # Loss function
    loss = "CosFace",          # Loss type: CosFace, ArcFace, etc.
    
    # Partial FC settings (for large datasets)
    sample_rate = 1.0,         # Sampling rate (1.0 = use all classes)
    
    # Output settings
    output = "work_dirs/ms1mv3_r50",  # Output directory
)
```

### Common Configuration Parameters

**Network Settings:**
- `network`: Backbone architecture (`r50`, `r100`, `r200`, `vit_b`, `vit_l`, etc.)
- `embedding_size`: Size of face embeddings (default: 512)
- `fp16`: Enable mixed precision training (True/False)

**Dataset Settings:**
- `rec`: Path to the .rec file
- `num_classes`: Number of identities (check dataset properties)
- `num_image`: Total images in dataset
- `num_workers`: Number of data loading threads (default: 8)

**Training Settings:**
- `batch_size`: Images per GPU per iteration
  - Larger = faster training but more memory
  - Typical values: 64-512 depending on GPU memory
- `num_epoch`: Total training epochs (typical: 20-40)
- `lr`: Learning rate
  - Default: 0.1 for batch size 512
  - Scale linearly with batch size

**Partial FC Settings** (for large datasets):
- `sample_rate`: Fraction of classes to sample per batch
  - 1.0 = all classes (full softmax)
  - 0.1 = 10% of classes (10x less memory)
  - Lower values = less memory, similar accuracy

### Creating Your Own Config

To create a custom configuration:

1. Copy an existing config:
   ```bash
   cp configs/ms1mv3_r50.py configs/my_custom_config.py
   ```

2. Edit the parameters:
   ```python
   # Change these based on your needs
   network = "r100"           # Use ResNet-100
   batch_size = 256          # Increase batch size
   num_epoch = 30            # Train longer
   lr = 0.2                  # Adjust learning rate
   ```

3. Run training with your config:
   ```bash
   torchrun --nproc_per_node=8 train_v2.py configs/my_custom_config.py
   ```

## Model Zoo

Pre-trained models are available for download. These models are trained on large-scale datasets and can be used for:
- Transfer learning
- Fine-tuning on your dataset
- Direct deployment for face recognition

### Download Pre-trained Models

**Option 1: Baidu Yun Pan**
- Link: [https://pan.baidu.com/s/1CL-l4zWqsI1oDuEEYVhj-g](https://pan.baidu.com/s/1CL-l4zWqsI1oDuEEYVhj-g)
- Password: `e8pw`

**Option 2: OneDrive**
- Link: [OneDrive Folder](https://1drv.ms/u/s!AswpsDO2toNKq0lWY69vN58GR6mw?e=p9Ov5d)

### Using Pre-trained Models

1. **Download a model** (e.g., `glint360k_r100.pth`)

2. **Load for inference:**
   ```python
   import torch
   
   # Load the model
   model = torch.load('glint360k_r100.pth')
   model.eval()
   
   # Extract features
   with torch.no_grad():
       features = model(input_images)
   ```

3. **Fine-tune on your dataset:**
   ```bash
   # Modify config to load pretrained weights
   python train_v2.py configs/my_config.py --pretrained glint360k_r100.pth
   ```

### Model Naming Convention

Models are named as: `[dataset]_[backbone]_[variant].pth`

Examples:
- `ms1mv3_r50.pth`: MS1MV3 dataset, ResNet-50
- `glint360k_r100.pth`: Glint360K dataset, ResNet-100
- `wf42m_vit_b.pth`: WebFace42M dataset, Vision Transformer Base

### License

**Important:** All pre-trained models are provided for **non-commercial research purposes only**. If you want to use them commercially, please contact the authors for licensing.

## Performance Benchmarks

### Evaluation Metrics

Models are evaluated on standard face recognition benchmarks:

**IJB-C (IARPA Janus Benchmark-C):**
- Large-scale unconstrained face recognition benchmark
- Reports True Accept Rate (TAR) at different False Accept Rates (FAR)
- TAR@FAR=1E-4 and TAR@FAR=1E-5 are common metrics

**MFR-ALL (Masked Face Recognition):**
- 242,143 identities, 1,624,305 images
- Measures TAR@FAR<1e-6

### Sample Results

#### Single-Host GPU Training (8x GPUs)

| Dataset | Backbone | MFR-ALL | IJB-C(1E-4) | IJB-C(1E-5) |
|---------|----------|---------|-------------|-------------|
| MS1MV3 | r50 | 79.14% | 96.37% | 94.47% |
| MS1MV3 | r100 | 81.97% | 96.85% | 95.02% |
| Glint360K | r100 | 89.52% | 97.55% | 96.38% |
| WF42M | r100 | 96.27% | 97.70% | 96.31% |
| WF42M | ViT-B | 97.16% | 97.91% | 97.05% |

#### Multi-Host GPU Training (16-64 GPUs)

| Dataset | Backbone | GPUs | MFR-ALL | Throughput |
|---------|----------|------|---------|-----------|
| WF42M | r50 | 16 | 93.96% | ~11K img/s |
| WF42M | r100 | 32 | 96.57% | ~9.8K img/s |
| WF42M | ViT-L | 64 | 97.85% | ~9.4K img/s |

### Training Speed Comparison

**Hardware:** 8x Tesla V100 32GB

**Partial FC vs. Traditional Methods:**

| Dataset Size | Data Parallel | Model Parallel | Partial FC |
|-------------|--------------|----------------|-----------|
| 125K IDs | 4,681 img/s | 4,824 img/s | **5,004 img/s** |
| 1.4M IDs | 1,672 img/s | 3,043 img/s | **4,738 img/s** |
| 5.5M IDs | Out of Memory | 1,389 img/s | **3,975 img/s** |
| 29M IDs | Out of Memory | Out of Memory | **1,855 img/s** |

**Key insight:** Partial FC enables training on datasets that would otherwise not fit in GPU memory.

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: CUDA Out of Memory

**Error message:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions:**
1. **Reduce batch size** in config file:
   ```python
   batch_size = 64  # Try smaller values: 32, 16
   ```

2. **Enable gradient checkpointing:**
   ```python
   checkpoint = True  # In config file
   ```

3. **Use Partial FC** for large datasets:
   ```python
   sample_rate = 0.1  # Sample only 10% of classes
   ```

4. **Use mixed precision training:**
   ```python
   fp16 = True  # In config file
   ```

#### Issue 2: Training is Very Slow

**Solutions:**
1. **Install NVIDIA DALI:**
   ```bash
   pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110
   ```

2. **Increase number of data workers:**
   ```python
   num_workers = 16  # In config file (default: 8)
   ```

3. **Use SSD instead of HDD** for dataset storage

4. **Reduce image size** if using very high-resolution images

#### Issue 3: Can't Find Dataset

**Error message:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/ms1m-retinaface-t1/train.rec'
```

**Solutions:**
1. **Check dataset path** in config file:
   ```python
   rec = "data/ms1m-retinaface-t1"  # Verify this path exists
   ```

2. **Use absolute path:**
   ```python
   rec = "/home/user/datasets/ms1m-retinaface-t1"
   ```

3. **Verify dataset structure:**
   ```bash
   ls -la data/ms1m-retinaface-t1/
   # Should show: train.rec, train.idx, property
   ```

#### Issue 4: Multi-Node Training Not Working

**Error message:**
```
RuntimeError: Connection refused or timeout
```

**Solutions:**
1. **Check network connectivity:**
   ```bash
   # On worker node, ping master node
   ping 192.168.1.100
   ```

2. **Check firewall settings:**
   ```bash
   # Allow the port on master node
   sudo ufw allow 12581
   ```

3. **Verify IP address:**
   ```bash
   # On master node
   ifconfig
   # Use the correct network interface IP
   ```

4. **Start all nodes within 60 seconds** of each other

#### Issue 5: Model Not Learning (Loss Not Decreasing)

**Solutions:**
1. **Check learning rate:**
   ```python
   lr = 0.1  # Try: 0.01, 0.001 if loss explodes
   ```

2. **Verify dataset labels** are correct

3. **Check if model is loaded correctly:**
   ```python
   # Add debug prints in train_v2.py
   print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
   ```

4. **Try a smaller, proven config first** (e.g., MS1MV3 + r50)

#### Issue 6: Installation Issues

**Problem: PyTorch not recognizing GPU**
```python
import torch
print(torch.cuda.is_available())  # Returns False
```

**Solutions:**
1. **Reinstall PyTorch with correct CUDA version:**
   ```bash
   # Check your CUDA version
   nvidia-smi
   
   # Install matching PyTorch version
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Check CUDA installation:**
   ```bash
   nvcc --version
   ```

## FAQ

### General Questions

**Q: How long does training take?**

A: Training time depends on:
- Dataset size
- Number of GPUs
- Model architecture

Typical times:
- MS1MV3 on 8x V100: ~4-6 hours (20 epochs)
- Glint360K on 8x V100: ~12-18 hours
- WebFace42M on 64x A100: ~2-3 days

**Q: How much GPU memory do I need?**

A: Minimum requirements:
- ResNet-50: 12GB per GPU (batch size 128)
- ResNet-100: 16GB per GPU (batch size 128)
- ViT-B: 20GB per GPU (batch size 128)
- ViT-L: 32GB per GPU (batch size 64)

**Q: Can I train on a single GPU?**

A: Yes, but:
- Training will be much slower
- You may need to reduce batch size
- Only practical for small datasets (MS1MV3 or smaller)
- Not recommended for production use

**Q: What's the difference between configs?**

A: Config files differ in:
- `_onegpu`: Optimized for single GPU (smaller batch)
- `_8gpu`: For 8 GPUs (standard)
- `_16gpu` / `_32gpu`: For multi-node training
- `pfc02` / `pfc03`: Different Partial FC sampling rates

**Q: Can I use AMD GPUs or CPU?**

A: No, this implementation requires:
- NVIDIA GPUs (CUDA support)
- Not compatible with AMD ROCm or CPU-only training

### Technical Questions

**Q: What is Partial FC?**

A: Partial FC (Partial Fully Connected layer) is a technique that:
- Samples only a subset of classes per training batch
- Dramatically reduces GPU memory usage
- Maintains similar accuracy to full softmax
- Enables training on datasets with millions of identities
- Is essential for WebFace42M and other large datasets

**Q: Should I use CosFace or ArcFace loss?**

A: Both work well:
- **ArcFace**: Slightly better accuracy, more stable training
- **CosFace**: Simpler, faster computation

Recommendation: Use ArcFace unless you have specific reasons to choose CosFace.

**Q: What backbone should I use?**

A: Choose based on your needs:
- **MobileFaceNet**: Fast inference, lower accuracy (mobile apps)
- **ResNet-50**: Best balance of speed and accuracy (recommended for most)
- **ResNet-100**: Better accuracy, moderate speed
- **ViT-B**: Best accuracy for face recognition
- **ViT-L**: Highest accuracy, slowest, needs most memory

**Q: How do I export to ONNX for deployment?**

A: Use the export script:
```bash
python tools/export_onnx.py --checkpoint work_dirs/model.pth --output model.onnx
```

**Q: Can I fine-tune on my own dataset?**

A: Yes:
1. Prepare your dataset in MXNet format
2. Load a pre-trained model
3. Train with a lower learning rate
4. Use fewer epochs (5-10)

**Q: What's the best learning rate?**

A: Default guidelines:
- Batch size 512: lr = 0.1
- Scale linearly with batch size
- For fine-tuning: use 10x smaller lr (0.01)
- If loss explodes: reduce by 10x

**Q: How do I monitor training?**

A: Three ways:
1. **Console output**: Real-time loss, accuracy, learning rate
2. **TensorBoard**: `tensorboard --logdir work_dirs`
3. **Log files**: Saved in `work_dirs/[config_name]/`

## Best Practices

### For Beginners
1. Start with **MS1MV3 dataset** and **ResNet-50** backbone
2. Use **single node, 8 GPUs** if available
3. Don't modify hyperparameters initially
4. Monitor TensorBoard to understand training dynamics
5. Save checkpoints frequently

### For Advanced Users
1. Use **Partial FC** for datasets > 1M identities
2. Enable **NVIDIA DALI** for 2-3x faster data loading
3. Tune **learning rate** based on batch size
4. Use **gradient checkpointing** for larger models
5. Experiment with **ViT backbones** for best accuracy

### Performance Optimization
1. **Use NVMe SSDs** for dataset storage
2. **Increase num_workers** if CPU is not saturated
3. **Enable fp16** (mixed precision) for faster training
4. **Use larger batch sizes** if GPU memory allows
5. **Pin memory** in data loader for faster host-to-GPU transfer

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{deng2019arcface,
  title={ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
  author={Deng, Jiankang and Guo, Jia and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={4690--4699},
  year={2019}
}

@inproceedings{an2022partialfc,
  title={Killing Two Birds with One Stone: Efficient and Robust Training of Face Recognition CNNs by Partial FC},
  author={An, Xiang and Deng, Jiankang and Guo, Jia and Feng, Ziyong and Zhu, XuHan and Yang, Jing and Liu, Tongliang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={4176--4185},
  year={2022}
}

@inproceedings{zhu2021webface260m,
  title={WebFace260M: A Benchmark Unveiling the Power of Million-Scale Deep Face Recognition},
  author={Zhu, Zheng and Huang, Guan and Deng, Jiankang and Ye, Yun and Huang, Junjie and Chen, Xinze and Zhu, Jiagang and Yang, Tian and Lu, Jiwen and Du, Dalong and Zhou, Jie},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={10492--10502},
  year={2021}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

**Important:** The pre-trained models are provided for **non-commercial research purposes only**. For commercial use, please contact the authors for licensing.

## Acknowledgments

This project is part of the [InsightFace](https://github.com/deepinsight/insightface) project.

**Maintainer:** [@anxiangsir](https://github.com/anxiangsir)

**Contributors:** We thank all contributors to the InsightFace project.

## Support

If you encounter any issues:

1. **Check the FAQ section** above
2. **Search existing issues** on GitHub
3. **Open a new issue** with:
   - Your system configuration
   - Complete error message
   - Steps to reproduce
   - What you've already tried

For questions about the paper or algorithm, please refer to the original publications.

---

**Happy Training! 祝训练顺利！**
