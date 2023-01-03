# Installation
## Prerequisites

1. Linux x64.
2. NVIDIA Driver supporting CUDA 10.0 or later (i.e., 410.48 or later driver releases).
3. (Optional) One or more of the following deep learning frameworks:

    * [MXNet 1.3](http://mxnet.incubator.apache.org/) `mxnet-cu100` or later.
    * [PyTorch 0.4](https://pytorch.org/) or later.
    * [TensorFlow 1.7](https://www.tensorflow.org/) or later.

## DALI in NGC Containers 
DALI is preinstalled in the TensorFlow, PyTorch, and MXNet containers in versions 18.07 and later on NVIDIA GPU Cloud.

## pip - Official Releases

### nvidia-dali

Execute the following command to install the latest DALI for specified CUDA version (please check support matrix to see if your platform is supported):

* For CUDA 10.2:

    ```bash
    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda102
    ```

* For CUDA 11.0:

    ```bash 
    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110
    ```


> Note: CUDA 11.0 build uses CUDA toolkit enhanced compatibility. It is built with the latest CUDA 11.x toolkit while it can run on the latest, stable CUDA 11.0 capable drivers (450.80 or later). Using the latest driver may enable additional functionality. More details can be found in [enhanced CUDA compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#enhanced-compat-minor-releases).

> Note: Please always use the latest version of pip available (at least >= 19.3) and update when possible by issuing pip install –upgrade pip

### nvidia-dali-tf-plugin

DALI doesn’t contain prebuilt versions of the DALI TensorFlow plugin. It needs to be installed as a separate package which will be built against the currently installed version of TensorFlow:

* For CUDA 10.2:

    ```bash
    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-tf-plugin-cuda102
    ```

* For CUDA 11.0:

    ```bash
    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-tf-plugin-cuda110
    ```

Installing this package will install `nvidia-dali-cudaXXX` and its dependencies, if they are not already installed. The package `tensorflow-gpu` must be installed before attempting to install `nvidia-dali-tf-plugin-cudaXXX`.

> Note: The packages `nvidia-dali-tf-plugin-cudaXXX` and `nvidia-dali-cudaXXX` should be in exactly the same version. Therefore, installing the latest `nvidia-dali-tf-plugin-cudaXXX`, will replace any older `nvidia-dali-cudaXXX` version already installed. To work with older versions of DALI, provide the version explicitly to the `pip install` command.

### pip - Nightly and Weekly Releases¶

> Note: While binaries available to download from nightly and weekly builds include most recent changes available in the GitHub some functionalities may not work or provide inferior performance comparing to the official releases. Those builds are meant for the early adopters seeking for the most recent version available and being ready to boldly go where no man has gone before.

> Note: It is recommended to uninstall regular DALI and TensorFlow plugin before installing nightly or weekly builds as they are installed in the same path

#### Nightly Builds
To access most recent nightly builds please use flowing release channel:

* For CUDA 10.2:

    ```bash
    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/nightly --upgrade nvidia-dali-nightly-cuda102
    ```

    ```
    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/nightly --upgrade nvidia-dali-tf-plugin-nightly-cuda102
    ```

* For CUDA 11.0:

    ```bash
    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/nightly --upgrade nvidia-dali-nightly-cuda110
    ```

    ```bash
    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/nightly --upgrade nvidia-dali-tf-plugin-nightly-cuda110
    ```


#### Weekly Builds

Also, there is a weekly release channel with more thorough testing. To access most recent weekly builds please use the following release channel (available only for CUDA 11):

```bash
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/weekly --upgrade nvidia-dali-weekly-cuda110
```

```bash
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/weekly --upgrade nvidia-dali-tf-plugin-week
```


---

### For more information about Dali and installation, please refer to [DALI documentation](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html).
