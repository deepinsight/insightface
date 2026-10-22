# InspireFace

[![GitHub release](https://img.shields.io/github/v/release/HyperInspire/InspireFace.svg?style=flat&color=blue&label=GitHub+release&logo=github)](https://github.com/HyperInspire/InspireFace/releases/latest)
[![Model](https://img.shields.io/github/v/release/HyperInspire/InspireFace.svg?style=flat&color=blue&label=Model+Zoo&logo=github)](https://github.com/HyperInspire/InspireFace/releases/tag/v1.x)
[![PyPI](https://img.shields.io/pypi/v/inspireface.svg?style=flat&color=orange&label=PyPI+release&logo=python)](https://pypi.org/project/inspireface/)
[![JitPack](https://jitpack.io/v/HyperInspire/inspireface-android-sdk.svg)](https://jitpack.io/#HyperInspire/inspireface-android-sdk)
[![Release SDKs](https://github.com/HyperInspire/InspireFace/actions/workflows/release-sdks.yaml/badge.svg)](https://github.com/HyperInspire/InspireFace/actions/workflows/release-sdks.yaml)
[![Ubuntu x86 Tests](https://github.com/HyperInspire/InspireFace/actions/workflows/test_ubuntu_x86_Pikachu.yaml/badge.svg)](https://github.com/HyperInspire/InspireFace/actions/workflows/test_ubuntu_x86_Pikachu.yaml)
[![Documentation](https://img.shields.io/badge/Document-Building-blue?style=flat&logo=readthedocs)](https://doc.inspireface.online/)

InspireFace is a cross-platform face recognition SDK developed in C/C++, supporting multiple operating systems and various backend types for inference, such as CPU, GPU, and NPU.

If you require further information on tracking development branches, CI/CD processes, or downloading pre-compiled libraries, please visit our [development repository](https://github.com/HyperInspire/InspireFace).

Please contact [contact@insightface.ai](mailto:contact@insightface.ai?subject=InspireFace) using your company e-mail for commercial support, including obtaining and integrating higher accuracy models, as well as custom development.

<img src="images/banner.webp" alt="banner" style="zoom:80%;" />

---

📘 For detailed development and integration guides, see the [documentation](https://doc.inspireface.online/).

## Change Logs

**`2026-09-27`** Add Windows platform support and precompiled SDKs.

**`2026-09-20`** Add demos for passive face liveness detection and flash liveness detection.

**`2026-08-30`** Add HarmonyOS support and configurable best-frame face capture, improve SDK performance, reliability, API compatibility, and Python usability, and expand regression tests, benchmarks, and bilingual documentation.

**`2026-07-22`** Add an Android example app.

**`2025-08-03`** Add a multi-link model download channel for the Python-SDK.

**`2025-06-15`** The [ErrorCode-Table](/doc/Error-Feedback-Codes.md) has been reorganized and streamlined.

**`2025-06-08`** Add facial expression recognition.

**`2025-04-27`** Optimize some issues and provide a stable version.

**`2025-03-16`** Acceleration using NVIDIA-GPU (**CUDA**) devices is already supported.

**`2025-03-09`** Release of android sdk in JitPack.

**`2025-02-20`** Upgrade the face landmark model.

**`2025-01-21`** Update all models to t3 and add tool to convert cosine similarity to percentage.

**`2025-01-08`** Support inference on Rockchip devices **RK3566/RK3568** NPU.

**`2024-12-25`** Add support for optional **RKRGA** image acceleration processing on Rockchip devices.

**`2024-12-22`** Started adapting for multiple Rockchip devices with NPU support, beginning with **RV1103/RV1106** support.

**`2024-12-10`** Added support for quick installation via Python package manager.

**`2024-10-09`** Added system resource monitoring and session statistics.

**`2024-09-30`** Fixed some bugs in the feature hub.

**`2024-08-18`** Updating [Benchmark](doc/Benchmark-Remark(Updating).md): Using CoreML with Apple's Neural Engine (ANE) on the iPhone 13, the combined processes of **Face Detection** + **Alignment** + **Feature Extraction** take less than **2ms**.

**`2024-07-17`** Add global resource statistics monitoring to prevent memory leaks.

**`2024-07-07`** Add some face action detection to the face interaction module.

**`2024-07-05`** Fixed some bugs in the python ctypes interface.

**`2024-07-03`** Add the blink detection algorithm of face interaction module.

**`2024-07-02`** Fixed several bugs in the face detector with multi-level input.

## License

The licensing of the open-source models employed by InspireFace adheres to the same requirements as InsightFace, specifying their use solely for academic purposes and explicitly prohibiting commercial applications.

## Quick Start

### Try the Android Example App

- Add a passive face liveness detection demo.
- Add a flash liveness detection demo.

<p>
  <a href="http://fir.tunm.top/pro/pz7b3dgv">
    <img src="images/inspireface-android-example-app-download.png" alt="Scan the QR code to download the InspireFace Android example app" width="220" />
  </a>
</p>

<p>
  <strong>Scan the QR code to download the Android example app</strong><br />
  Or <a href="http://fir.tunm.top/pro/pz7b3dgv">download the app directly</a>.
</p>

### Python Quick Start

For Python users on **Windows x64, Linux, and macOS**, InspireFace can be quickly installed via pip. CPU wheels are available for Windows x64 and for Linux/macOS on x86_64 and ARM64:

```bash
pip install -U inspireface
```

The wheels include the native CPU library, and models are downloaded automatically on first use. On Windows, use x64 Python and install the Microsoft Visual C++ 2022 x64 Redistributable as described in the [Windows runtime requirements](doc/Windows.md#installed-cc-consumers). After installation, you can use inspireface like this:

```Python
import cv2
import inspireface as isf

# Create a session with optional features
opt = isf.HF_ENABLE_NONE
session = isf.InspireFaceSession(opt, isf.HF_DETECT_MODE_ALWAYS_DETECT)

# Load the image using OpenCV.
image = cv2.imread(image_path)

# Perform face detection on the image.
faces = session.face_detection(image)

for face in faces:
    x1, y1, x2, y2 = face.location
    rect = ((x1, y1), (x2, y2), face.roll)
     # Calculate center, size, and angle
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    size = (x2 - x1, y2 - y1)
    angle = face.roll

    # Apply rotation to the bounding box corners
    rect = ((center[0], center[1]), (size[0], size[1]), angle)
    box = cv2.boxPoints(rect)
    box = box.astype(int)

    # Draw the rotated bounding box
    cv2.drawContours(image, [box], 0, (100, 180, 29), 2)

cv2.imshow("face detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

⚠️The project is currently in a rapid iteration phase, **before each update**, please pull the latest model from the remote side!

```python
import inspireface

for model in ["Pikachu", "Megatron"]:
    inspireface.pull_latest_model(model)
```

More examples can be found in the [python](python/) directory.

## Preparation
### Clone 3rdparty

Clone the `3rdparty` repository from the remote repository into the root directory of the project. Note that this repository contains some submodules. When cloning, you should use the `--recurse-submodules` parameter, or after entering the directory, use `git submodule update --init --recursive` to fetch and synchronize the latest submodules:

```Bash
# Must enter this directory
cd InspireFace
# Clone the repository and pull submodules
git clone --recurse-submodules https://github.com/tunmx/inspireface-3rdparty.git 3rdparty
```

If you need to update the `3rdparty` repository to ensure it is current, or if you didn't use the `--recursive` parameter during the initial pull, you can run `git submodule update --init --recursive`:

```bash
# Must enter this directory
cd InspireFace
# If you're not using recursive pull
git clone https://github.com/tunmx/inspireface-3rdparty.git 3rdparty

cd 3rdparty
git pull
# Update submodules
git submodule update --init --recursive
```

### Downloading Model Package Files

You can download the model package files containing models and configurations needed for compilation from [Release Page](https://github.com/HyperInspire/InspireFace/releases/tag/v1.x) and extract them to any location. 

You can use the **command/download_models_general.sh** command to download resource files, which will be downloaded to the **test_res/pack** directory. This way, when running the Test program, it can access and read the resource files from this path by default.

⚠️The project is currently in a rapid iteration phase, **before each update**, please pull the latest model from the remote side!

```bash
# Download lightweight resource files for mobile device
bash command/download_models_general.sh Pikachu
# Download resource files for mobile device or PC/server
bash command/download_models_general.sh Megatron
# Download resource files for RV1109
bash command/download_models_general.sh Gundam_RV1109
# Download resource files for RV1106
bash command/download_models_general.sh Gundam_RV1106
# Download resource files for RK356X
bash command/download_models_general.sh Gundam_RK356X
# Download resource files for RK3588
bash command/download_models_general.sh Gundam_RK3588
# Download resource files for NVIDIA-GPU Device(TensorRT)
bash command/download_models_general.sh Megatron_TRT

# Download all model files
bash command/download_models_general.sh
```

### Installing MNN
The '**3rdparty**' directory already includes the MNN library and specifies a particular version as the stable version. If you need to enable or disable additional configuration options during compilation, you can refer to the CMake Options provided by MNN. If you need to use your own precompiled version, feel free to replace it.

### Requirements

- CMake (version 3.20 or higher)
- Ninja (required by the Windows build scripts; included in Visual Studio's **C++ CMake tools for Windows** component)
- Git (to fetch the source and recursive submodules)
- 64-bit PowerShell (required to run the Windows build scripts)
- Python 3.7 or newer with pip (for Python wheel builds; use x64 Python on Windows) [**Optional**]
- NDK (version 16 or higher, only required for Android) [**Optional**]
- MNN (version 3.x or higher)
- C++ Compiler
    - On Linux/macOS, use GCC or Clang (macOS provides Clang through Xcode or the Xcode Command Line Tools)
        - Recommended GCC version is 4.9 or higher
            - Note that in some distributions, GCC (GNU C Compiler) and G++ (GNU C++ Compiler) are installed separately.
            - For instance, on Ubuntu, you need to install both gcc and g++
        - Recommended Clang version is 3.9 or higher
    - MSVC v143 from Visual Studio 2022 or Build Tools 2022 (Windows x64)
        - Install the **Desktop development with C++** workload, including the MSVC x64/x86 build tools and a Windows 10 or Windows 11 SDK.
    - arm-linux-gnueabihf (for RV1109/RV1126) [**Optional**]
        - Prepare the cross-compilation toolchain in advance, such as gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf
- CUDA (version 11.x or higher) [**Optional**]
    - GPU-based inference requires installing NVIDIA's CUDA dependencies on the device.
- TensorRT (version 10 or higher) [**Optional**]
- Eigen3

- RKNN [**Optional**]
    - Adjust and select versions currently supported for specific requirements.

## Compilation
CMake option are used to control the various details of the compilation phase. Please select according to your actual requirements. [CMake Option](doc/CMake-Option.md).

For Windows x64 desktop and Windows Server builds with the CPU/MNN backend, use the PowerShell entry point
[`command/build_windows.ps1`](command/build_windows.ps1) from a Visual Studio
x64 Native Tools environment. See [Windows build and SDK consumption](doc/Windows.md)
for test resources, shared/static builds, and the installed CMake package.

### Local Compilation
If you are using macOS or Linux, you can quickly compile using the shell scripts provided in the `command` folder at the project root:
```bash
cd InspireFace/
# Execute the local compilation script
bash command/build.sh
```
On Windows, run `.\command\build_windows.ps1` from PowerShell in a Visual Studio x64 Native Tools environment. The default Release shared SDK is installed to `build/windows-x64-Release-shared/install/InspireFace`; its `lib/` directory contains `libInspireFace.dll` and the import library. Add `-Static` for a static SDK. Test resources must be prepared before using the default build, which also runs CTest; see the [Windows build instructions](doc/Windows.md).

Compilation results are placed in the build directory. For example, a Linux shared SDK has the following layout:
```bash
inspireface-linux
   ├── include
   │   ├── herror.h
   │   ├── intypedef.h
   │   ├── inspireface.h
   │   ├── inspirecv/
   │   └── inspireface/
   └── lib
       └── libInspireFace.so
```

- **libInspireFace.so**：Compiled dynamic linking library.
- **inspireface.h**：Header file definition.
- **herror.h**：Reference error number definition.
- **intypedef.h**: Type definition file.
- **inspirecv**: Simple cv library CPP header file folder.
- **inspireface**: inspireface cpp header folder.
### Cross Compilation
Cross compilation requires you to prepare the target platform's cross-compilation toolchain on the host machine in advance. Here, compiling for Rockchip's embedded devices RV1106 is used as an example:
```bash
# Set the path for the cross-compilation toolchain
export ARM_CROSS_COMPILE_TOOLCHAIN=YOUR_DIR/arm-rockchip830-linux-uclibcgnueabihf
# Execute the cross-compilation script for RV1106
bash command/build_cross_rv1106_armhf_uclibc.sh
```
After the compilation is complete, you can find the compiled results in the `build/inspireface-linux-armv7-rv1106-armhf-uclibc` directory.

### iOS Compilation

To compile for iOS, ensure you are using a Mac device. The script will automatically download third-party dependencies into the `.macos_cache` directory.

```
bash command/build_ios.sh
```

After the compilation is complete, `inspireface.framework` will be placed in the `build/inspireface-ios` directory.

### Android Compilation

With a JDK (8 or later), CMake and Python 3 installed, set your Android NDK path and build:

```
export ANDROID_NDK=YOUR_ANDROID_NDK_PATH
bash command/build_android.sh
```

The `build/inspireface-android` directory contains arm64-v8a, armeabi-v7a and x86_64 libraries, C/C++ headers, and `java/inspireface.jar`. Each ABI directory contains a single `libInspireFace.so` with the C/C++ API, legacy Android JNI and complete Java binding.

### HarmonyOS Compilation and ArkTS Usage

Prepare an OpenHarmony Native SDK, set its native directory, and build the ARM64 Node-API module:

```bash
export OHOS_NATIVE_HOME=/path/to/openharmony-sdk/native
./command/build_harmonyos_napi.sh
```

The build stages an importable HAR module at `build/inspireface-harmonyos-napi-arm64-v8a/install/HarmonyOS/har`. Add this module to a DevEco Studio project, then use the ArkTS wrapper as follows. `modelPath` must be a filesystem path accessible to the application, and the image byte length must exactly match its format and dimensions.

```typescript
import { DetectMode, ImageFormat, InspireFace, Rotation } from '@hyperinspire/inspireface';

InspireFace.launch(modelPath);
const session = InspireFace.createSession({
  detectMode: DetectMode.ALWAYS_DETECT,
  maxFaces: 5
});
const image = InspireFace.createImageStream(rgbaBytes, width, height,
  ImageFormat.RGBA, Rotation.DEGREE_0);

try {
  const result = session.track(image);
  try {
    console.info(`Detected faces: ${result.detectedNum}`);
  } finally {
    session.releaseFaceResult(result);
  }
} finally {
  image.close();
  session.close();
  InspireFace.terminate();
}
```

The ArkTS package covers the complete portable C API surface, including Pipeline, FeatureHub, image bitmaps, aligned features, diagnostics, and hardware capability queries. Platform-inapplicable backends such as CUDA and CoreML remain callable for parity and report `UNSUPPORTED`. For ownership and complete API notes, see [HarmonyOS SDK documentation](harmony/inspireface/README.md).

### Linux-based NVIDIA GPU Acceleration with TensorRT Compilation

If you want to use NVIDIA GPU devices for accelerated inference on Linux, you need to install **CUDA**, **cuDNN**, and **TensorRT-10** on your device, and configure the relevant environment variables.

```bash
# Example, Change to your TensorRT-10 path
export TENSORRT_ROOT=/user/tunm/software/TensorRT-10
```

Before compiling, please ensure that your related environments such as CUDA and TensorRT-10 are available. If you encounter issues with finding CUDA libraries during the compilation process, you may need to check whether the relevant environment variables have been configured: `CUDA_TOOLKIT_ROOT_DIR`, `CUDA_CUDART_LIBRARY`.

```bash
bash command/build_linux_tensorrt.sh
```

Additionally, you can use **NVIDIA's Docker images** for compilation. For example, to compile using a **CUDA 12** and **TensorRT-10** image on Ubuntu 22.04, you can execute the following commands:

```bash
docker-compose up build-tensorrt-cuda12-ubuntu22
```

If you want to use pre-compiled libraries, you can use **[FindTensorRT.cmake](toolchain/FindTensorRT.cmake)** to create links to CUDA and TensorRT.

### React Native

For Android and iOS, in addition to the native interface, you can use the React Native library powered by Nitro Modules and JSI—providing ultra-fast, seamless bindings to the InspireFace SDK. For more details, check out the [react-native-nitro-inspire-face](https://github.com/ronickg/react-native-nitro-inspire-face) repository or the [documentation](https://ronickg.github.io/react-native-nitro-inspire-face). Author: ronickg.

### Supported Platforms and Architectures

We have completed the adaptation and testing of the software across various operating systems and CPU architectures. This includes compatibility verification for platforms such as Windows x64, Linux, macOS, iOS, and Android, as well as testing for specific hardware support to ensure stable operation in diverse environments. Windows desktop and Windows Server builds currently support the x64 CPU/MNN backend.

| No. | Platform | Architecture<sup><br/>(CPU) | Device<sup><br/>(Special) | **Supported** | Passed Tests | Release<sup><br/>(Online) |
| ------- | -------------------- | --------------------- | -------------------------- | :-----------: | :----------------: | :----------------: |
| 1       | **Linux**<sup><br/>(CPU)      | ARMv7                 | -                          | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![build](https://img.shields.io/github/actions/workflow/status/HyperInspire/InspireFace/release-sdks.yaml?label=✓&labelColor=success&color=success&failedLabel=✗&failedColor=critical&logo=github&logoColor=white)](https://github.com/HyperInspire/InspireFace/actions/workflows/release-sdks.yaml) |
| 2       |                      | ARMv8                 | -                          | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![build](https://img.shields.io/github/actions/workflow/status/HyperInspire/InspireFace/release-sdks.yaml?label=✓&labelColor=success&color=success&failedLabel=✗&failedColor=critical&logo=github&logoColor=white)](https://github.com/HyperInspire/InspireFace/actions/workflows/release-sdks.yaml) |
| 3       |                      | x86/x86_64            | -                          | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![build](https://img.shields.io/github/actions/workflow/status/HyperInspire/InspireFace/release-sdks.yaml?label=✓&labelColor=success&color=success&failedLabel=✗&failedColor=critical&logo=github&logoColor=white)](https://github.com/HyperInspire/InspireFace/actions/workflows/release-sdks.yaml) |
| 4       | **Linux**<sup><br/>(Rockchip) | ARMv7                 | RV1109/RV1126              | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![build](https://img.shields.io/github/actions/workflow/status/HyperInspire/InspireFace/release-sdks.yaml?label=✓&labelColor=success&color=success&failedLabel=✗&failedColor=critical&logo=github&logoColor=white)](https://github.com/HyperInspire/InspireFace/actions/workflows/release-sdks.yaml) |
| 5 | | ARMv7 | RV1103/RV1106 | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![build](https://img.shields.io/github/actions/workflow/status/HyperInspire/InspireFace/release-sdks.yaml?label=✓&labelColor=success&color=success&failedLabel=✗&failedColor=critical&logo=github&logoColor=white)](https://github.com/HyperInspire/InspireFace/actions/workflows/release-sdks.yaml) |
| 6 | | ARMv8 | RK3566/RK3568 | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![build](https://img.shields.io/github/actions/workflow/status/HyperInspire/InspireFace/release-sdks.yaml?label=✓&labelColor=success&color=success&failedLabel=✗&failedColor=critical&logo=github&logoColor=white)](https://github.com/HyperInspire/InspireFace/actions/workflows/release-sdks.yaml) |
| 7 | | ARMv8 | RK3588 | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | - | [![build](https://img.shields.io/github/actions/workflow/status/HyperInspire/InspireFace/release-sdks.yaml?label=✓&labelColor=success&color=success&failedLabel=✗&failedColor=critical&logo=github&logoColor=white)](https://github.com/HyperInspire/InspireFace/actions/workflows/release-sdks.yaml) |
| 8      | **Linux**<sup><br/>(MNN_CUDA) | x86/x86_64            | NVIDIA-GPU          | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | - |
| 9      | **Linux**<sup><br/>(CUDA) | x86/x86_64            | NVIDIA-GPU          | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![build](https://img.shields.io/github/actions/workflow/status/HyperInspire/InspireFace/release-sdks.yaml?label=✓&labelColor=success&color=success&failedLabel=✗&failedColor=critical&logo=github&logoColor=white)](https://github.com/HyperInspire/InspireFace/actions/workflows/release-sdks.yaml) |
| 10      | **MacOS**           | Intel       | CPU/Metal/**ANE** | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![build](https://img.shields.io/github/actions/workflow/status/HyperInspire/InspireFace/release-sdks.yaml?label=✓&labelColor=success&color=success&failedLabel=✗&failedColor=critical&logo=github&logoColor=white)](https://github.com/HyperInspire/InspireFace/actions/workflows/release-sdks.yaml) |
| 11   |                      | Apple Silicon         | -                          | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![build](https://img.shields.io/github/actions/workflow/status/HyperInspire/InspireFace/release-sdks.yaml?label=✓&labelColor=success&color=success&failedLabel=✗&failedColor=critical&logo=github&logoColor=white)](https://github.com/HyperInspire/InspireFace/actions/workflows/release-sdks.yaml) |
| 12     | **iOS**              | ARM                   | CPU/Metal/**ANE**         | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![build](https://img.shields.io/github/actions/workflow/status/HyperInspire/InspireFace/release-sdks.yaml?label=✓&labelColor=success&color=success&failedLabel=✗&failedColor=critical&logo=github&logoColor=white)](https://github.com/HyperInspire/InspireFace/actions/workflows/release-sdks.yaml) |
| 13     | **Android**          | ARMv7                 | -                          | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![build](https://img.shields.io/github/actions/workflow/status/HyperInspire/InspireFace/release-sdks.yaml?label=✓&labelColor=success&color=success&failedLabel=✗&failedColor=critical&logo=github&logoColor=white)](https://github.com/HyperInspire/InspireFace/actions/workflows/release-sdks.yaml) |
| 14     |                      | ARMv8                 | -                          | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![build](https://img.shields.io/github/actions/workflow/status/HyperInspire/InspireFace/release-sdks.yaml?label=✓&labelColor=success&color=success&failedLabel=✗&failedColor=critical&logo=github&logoColor=white)](https://github.com/HyperInspire/InspireFace/actions/workflows/release-sdks.yaml) |
| 15 | | x86_64 | - | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![build](https://img.shields.io/github/actions/workflow/status/HyperInspire/InspireFace/release-sdks.yaml?label=✓&labelColor=success&color=success&failedLabel=✗&failedColor=critical&logo=github&logoColor=white)](https://github.com/HyperInspire/InspireFace/actions/workflows/release-sdks.yaml) |
| 16 | **Android**<sup><br/>(Rockchip) | ARMv8 | RK3566/RK3568 | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![build](https://img.shields.io/github/actions/workflow/status/HyperInspire/InspireFace/release-sdks.yaml?label=✓&labelColor=success&color=success&failedLabel=✗&failedColor=critical&logo=github&logoColor=white)](https://github.com/HyperInspire/InspireFace/actions/workflows/release-sdks.yaml) |
| 17 |  | ARMv8 | RK3588 | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![build](https://img.shields.io/github/actions/workflow/status/HyperInspire/InspireFace/release-sdks.yaml?label=✓&labelColor=success&color=success&failedLabel=✗&failedColor=critical&logo=github&logoColor=white)](https://github.com/HyperInspire/InspireFace/actions/workflows/release-sdks.yaml) |
| 18 | **HarmonyOS** | ARMv8 | - | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | - | [![build](https://img.shields.io/github/actions/workflow/status/HyperInspire/InspireFace/release-sdks.yaml?label=✓&labelColor=success&color=success&failedLabel=✗&failedColor=critical&logo=github&logoColor=white)](https://github.com/HyperInspire/InspireFace/actions/workflows/release-sdks.yaml) |
| 19 | **Windows** | x86_64 (x64) | - | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![build](https://img.shields.io/github/actions/workflow/status/HyperInspire/InspireFace/release-sdks.yaml?label=✓&labelColor=success&color=success&failedLabel=✗&failedColor=critical&logo=github&logoColor=white)](https://github.com/HyperInspire/InspireFace/actions/workflows/release-sdks.yaml) |
| 20 |  | ARMv8 | - | - | - | - |
| 21 | **Linux**<sup><br/>(Jetson series) | ARMv8 | Jetson series | - | - | - |

- **Device**: Some special device support, primarily focused on computing power devices.
- **Supported**: The solution has been fully developed and successfully verified on offline devices.
- **Passed Tests**: The feature has at least **passed unit tests** on offline devices.
- **Release**: Links to the platform's [SDK release workflow](https://github.com/HyperInspire/InspireFace/actions/workflows/release-sdks.yaml) or published Python wheels on PyPI.

> HarmonyOS currently marks build and API adaptation support only. Passed Tests and Release remain unmarked until validation on HarmonyOS hardware is available.

### Multi-platform compilation using Docker

We offer a method for rapid multi-platform compilation using Docker, provided that Docker is installed beforehand, and the appropriate commands are executed:
```Bash
# Build x86 Ubuntu18.04
docker-compose up build-ubuntu18

# Build armv7 cross-compile
docker-compose up build-cross-armv7-armhf

# Build armv7 with support RV1109RV1126 device NPU cross-complie
docker-compose up build-cross-rv1109rv1126-armhf

# Build armv7 with support RV1106 device NPU cross-complie
docker-compose up build-cross-rv1106-armhf-uclibc

# Build armv8 with support RK356x device NPU cross-complie
docker-compose up build-cross-rk356x-aarch64

# Build Android with support arm64-v8a and armeabi-v7a
docker-compose up build-cross-android

# Compile the tensorRT back-end based on CUDA12 and then Ubuntu22.04
docker-compose up build-tensorrt-cuda12-ubuntu22

# Build all
docker-compose up
```

## Example
### C/C++ Sample: Use the recommended CAPI Interface
To integrate InspireFace into a C/C++ project, you simply need to link the InspireFace library and include the appropriate header files(We recommend using the more compatible **CAPI** headers). Below is a basic example demonstrating face detection:

```c
#include <inspireface.h>
#include <herror.h>

...
  
HResult ret;
// The resource file must be loaded before it can be used
ret = HFLaunchInspireFace(packPath);
if (ret != HSUCCEED) {
    HFLogPrint(HF_LOG_ERROR, "Load Resource error: %d", ret);
    return ret;
}

// Enable the functions in the pipeline: mask detection, live detection, and face quality
// detection
HOption option = HF_ENABLE_QUALITY | HF_ENABLE_MASK_DETECT | HF_ENABLE_LIVENESS;
// Non-video or frame sequence mode uses IMAGE-MODE, which is always face detection without
// tracking
HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
// Maximum number of faces detected
HInt32 maxDetectNum = 20;
// Face detection image input level
HInt32 detectPixelLevel = 160;
// Handle of the current face SDK algorithm context
HFSession session = {0};
ret = HFCreateInspireFaceSessionOptional(option, detMode, maxDetectNum, detectPixelLevel, -1, &session);
if (ret != HSUCCEED) {
    HFLogPrint(HF_LOG_ERROR, "Create FaceContext error: %d", ret);
    return ret;
}

// Configure some detection parameters
HFSessionSetTrackPreviewSize(session, detectPixelLevel);
HFSessionSetFilterMinimumFacePixelSize(session, 4);

// Load a image
HFImageBitmap image;
ret = HFCreateImageBitmapFromFilePath(sourcePath, 3, &image);
if (ret != HSUCCEED) {
    HFLogPrint(HF_LOG_ERROR, "The source entered is not a picture or read error.");
    return ret;
}
// Prepare an image parameter structure for configuration
HFImageStream imageHandle = {0};
ret = HFCreateImageStreamFromImageBitmap(image, rotation_enum, &imageHandle);
if (ret != HSUCCEED) {
    HFLogPrint(HF_LOG_ERROR, "Create ImageStream error: %d", ret);
    return ret;
}

// Execute HF_FaceContextRunFaceTrack captures face information in an image
HFMultipleFaceData multipleFaceData = {0};
ret = HFExecuteFaceTrack(session, imageHandle, &multipleFaceData);
if (ret != HSUCCEED) {
    HFLogPrint(HF_LOG_ERROR, "Execute HFExecuteFaceTrack error: %d", ret);
    return ret;
}

// Print the number of faces detected
auto faceNum = multipleFaceData.detectedNum;
HFLogPrint(HF_LOG_INFO, "Num of face: %d", faceNum);

// The memory must be freed at the end of the program
ret = HFReleaseImageBitmap(image);
if (ret != HSUCCEED) {
    HFLogPrint(HF_LOG_ERROR, "Release image bitmap error: %d", ret);
    return ret;
}

ret = HFReleaseImageStream(imageHandle);
if (ret != HSUCCEED) {
    HFLogPrint(HF_LOG_ERROR, "Release image stream error: %d", ret);
}

ret = HFReleaseInspireFaceSession(session);
if (ret != HSUCCEED) {
    HFLogPrint(HF_LOG_ERROR, "Release session error: %d", ret);
    return ret;
}

...
```
For more examples, you can refer to the `cpp/sample` sub-project located in the root directory. You can compile these sample executables by enabling the `ISF_BUILD_WITH_SAMPLE` option during the compilation process.

**Note**: For each error code feedback, you can click on this [link](doc/Error-Feedback-Codes.md) to view detailed explanations.

### C++ Sample: Use the C++ version of the header files

If you want to use C++ header files, then you need to enable **ISF_INSTALL_CPP_HEADER** during compilation. Executing the install command will add the C++ header files.

```c++
#include <iostream>
#include <memory>
#include <inspireface/inspireface.hpp>

...

// Set log level to info
INSPIRE_SET_LOG_LEVEL(inspire::LogLevel::ISF_LOG_INFO);

int32_t ret = 0;
// Global init(you only need to call once)
ret = INSPIREFACE_CONTEXT->Load("Pikachu");
INSPIREFACE_CHECK_MSG(ret == HSUCCEED, "Load model failed");

// Create face algorithm session
inspire::ContextCustomParameter custom_param;
custom_param.enable_recognition = true;
auto max_detect_face = 5;
auto detect_level_px = 320; // 160, 320, 640

// Create a face algorithm session
std::shared_ptr<inspire::Session> session(
    inspire::Session::CreatePtr(inspire::DETECT_MODE_ALWAYS_DETECT, max_detect_face, custom_param, detect_level_px));

// Load image(default format is BGR)
inspirecv::Image image = inspirecv::Image::Create("face.jpg");

// Create frame process
inspirecv::FrameProcess process =
    inspirecv::FrameProcess::Create(image, inspirecv::BGR, inspirecv::ROTATION_0);

// Detect face
std::vector<inspire::FaceTrackWrap> detect_results;
ret = session->FaceDetectAndTrack(process, detect_results);
INSPIRE_LOGI("Number of faces detected: %d", detect_results.size());
if (detect_results.size() == 0)
{
    INSPIRE_LOGW("No face detected");
    return -1;
}

// Copy image
inspirecv::Image image_copy = image.Clone();
// Draw face
auto thickness = 2;
for (auto &face : detect_results)
{
    auto rect = session->GetFaceBoundingBox(face);
    auto lmk = session->GetNumOfFaceDenseLandmark(face);
    image_copy.DrawRect(rect, inspirecv::Color::Red, thickness);
    for (auto &point : lmk)
    {
        image_copy.DrawCircle(point.As<int>(), 0, inspirecv::Color::Orange, thickness);
    }
}
// Save draw image
image_copy.Write("result.jpg");

// Face Embedding extract
inspire::FaceEmbedding face_embedding;
// Extract the first face feature
ret = session->FaceFeatureExtract(process, detect_results[0], face_embedding);
INSPIRE_LOGI("Length of face embedding: %d", face_embedding.embedding.size());

...
```

Please note that the C++ interface has not been fully tested. It is recommended to use the **CAPI** interface as the primary option.

**More detailed cases**:

- [C Sample](cpp/sample/api/)
- [C++ Sample](cpp/sample/cpp_api/)

### Objective-C and Swift Samples (iOS / macOS)

Add `InspireFace.xcframework` to your Xcode project; Swift also requires `InspireFaceSwift.xcframework`. See the [Apple integration guide](https://doc.inspireface.online/zh/using-with/apple.html) for project setup and creating image streams from files or camera buffers.

These one-shot examples take a model resource file path (such as `Pikachu`) and an existing image stream, and return the number of detected faces. Models are provided separately from the frameworks; see [model paths and error handling](https://doc.inspireface.online/zh/using-with/apple.html#model-paths-and-errors).

#### Objective-C

```objective-c
#import <InspireFace/InspireFaceApple.h>

BOOL DetectFaces(NSString *modelPath, IFImageStream *stream,
                 HInt32 *faceCount, NSError **error) {
    *faceCount = 0;
    if (![IFRuntime launchAtPath:modelPath error:error]) return NO;
    IFSession *session = [[IFSession alloc] initWithOptions:0
                                                     mode:HF_DETECT_MODE_ALWAYS_DETECT
                                             maximumFaces:10
                                               pixelLevel:320
                                          framesPerSecond:-1
                                                    error:error];
    HFMultipleFaceData faces = {0};
    BOOL success = session && [session trackStream:stream borrowedResult:&faces error:error];
    if (success) *faceCount = faces.detectedNum;
    [session closeWithError:NULL];
    [IFRuntime terminateWithError:NULL];
    return success;
}
```

#### Swift

```swift
import InspireFaceSwift

func detectFaces(modelPath: String, stream: ImageStream) throws -> Int {
    try InspireFaceRuntime.launch(path: modelPath)
    defer { try? InspireFaceRuntime.terminate() }
    let session = try FaceSession(configuration: SessionConfiguration(
        detectionMode: .alwaysDetect, maximumFaces: 10, pixelLevel: 320))
    defer { try? session.close() }
    return try session.withUnsafeFaces(in: stream) { faces in
        faces.count
    }
}
```

Keep the input stream and its pixel storage alive during the call, then close the stream when finished. For continuous processing, initialize the runtime once and reuse a session on a serial work queue across frames.

### Python Native Sample

The Python implementation is compiled based on InspireFace source code, and is integrated using a native interface approach.

#### Use pip to install InspireFace

You can use pip to install the InspireFace Python package on Windows x64, Linux, and macOS:

```bash
pip install inspireface
```

#### Python Native Sample

For source-based Python development, copy or symlink the compiled native library into `python/inspireface/modules/core/libs/PLATFORM/ARCH/`, matching the Python process's OS and architecture. Use `linux` or `darwin` with `x64` or `arm64`, or `windows/x64` for Windows. The library is named `libInspireFace.so` on Linux, `libInspireFace.dylib` on macOS, and `libInspireFace.dll` on Windows. PyPI wheels already include this library. You can then start testing by navigating to the **[python](python/)** directory. Your Python environment will need to have some dependencies installed:

- python >= 3.7
- opencv-python
- loguru
- tqdm
- numpy
- ctypes
```bash
# Linux example: use a symbolic link to the matching architecture directory
mkdir -p python/inspireface/modules/core/libs/linux/x64
ln -s /absolute/path/to/YOUR_BUILD_DIR/install/InspireFace/lib/libInspireFace.so python/inspireface/modules/core/libs/linux/x64/
# Navigate to the sub-project directory
cd python
```

To build a Windows wheel, run `.\command\build_wheel_windows.ps1 -PythonExecutable python` from the repository root in an x64 Native Tools PowerShell environment. The wheel is written to `python/dist/inspireface-<version>-py3-none-win_amd64.whl`; see [Windows wheel builds](doc/Windows.md#python-and-platform-scope).

Import inspireface for a quick facial detection example:
```python
import cv2
import inspireface as isf

# Step 1: Initialize the SDK globally (only needs to be called once per application)
ret = isf.reload()
assert ret, "Launch failure. Please ensure the resource path is correct."

# Optional features, loaded during session creation based on the modules specified.
opt = isf.HF_ENABLE_FACE_POSE
session = isf.InspireFaceSession(opt, isf.HF_DETECT_MODE_ALWAYS_DETECT)

# Load the image using OpenCV.
image = cv2.imread(image_path)
assert image is not None, "Please check that the image path is correct."

# Perform face detection on the image.
faces = session.face_detection(image)
print(f"face detection: {len(faces)} found")

# Copy the image for drawing the bounding boxes.
draw = image.copy()
for idx, face in enumerate(faces):
    print(f"{'==' * 20}")
    print(f"idx: {idx}")
    # Print Euler angles of the face.
    print(f"roll: {face.roll}, yaw: {face.yaw}, pitch: {face.pitch}")
    # Draw bounding box around the detected face.
    x1, y1, x2, y2 = face.location
    cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 0, 255), 2)
```
In the project, more usage examples are provided:

- `sample_face_detection.py`: Facial detection example
- `sample_face_recognition.py`: Facial recognition example
- `sample_face_track_from_video.py`: Facial tracking from video stream example

### Java Sample (JVM, non-Android)

The portable Java binding provides the complete C API through JNI, with a Java 8-compatible JAR and native libraries built for the target OS and CPU architecture. Users do not need to write JNI code.

Build with a JDK, CMake and Python 3 installed:

```bash
bash command/build_java.sh
# To also run the JVM contract tests with the local test models:
# ISF_JAVA_TESTS=ON bash command/build_java.sh
```

The SDK is installed in `build/java-sdk/install/Java`, containing `inspireface.jar`, generated Java sources, an API manifest, native libraries and a [face detection example](java/examples/DetectFaces.java). For example, on macOS arm64:

```bash
cd build/java-sdk/install/Java
javac -cp inspireface.jar examples/DetectFaces.java
java -Djava.library.path=native/macos-arm64 -cp inspireface.jar:examples \
    DetectFaces /path/to/Pikachu /path/to/face.jpg
```

Choose the native directory matching the running JVM's OS and architecture. The JAR is shared across targets; the JNI library and its core SDK dependency must come from the same build. Model files are supplied separately. The low-level API uses `Native`, `NativeTypes` and `NativeConstants` in `com.insightface.sdk.inspireface.jni`; `InspireFaceException.check(status)` optionally converts C error codes into exceptions.

Image and feature buffers use direct `ByteBuffer` storage in native byte order. Input pixels are retained until the stream is released or its buffer is replaced. Borrowed output buffers retain the C API's validity limits; close native resources explicitly and serialize operations on each resource. The existing Android API remains available separately below.

CPU inference defaults to `NORMAL`. Configure the process-wide policy before creating sessions (the same API works on Android):

```java
import com.insightface.sdk.inspireface.jni.CPUEngine;

CPUEngine.setGlobalPowerMode(CPUEngine.PowerMode.NORMAL); // NORMAL, HIGH or LOW
CPUEngine.PowerMode mode = CPUEngine.getGlobalPowerMode();
```

The setting applies to subsequently initialized CPU runtimes and survives launch/reload/terminate. Existing runtimes, model thread counts and precision are unchanged. Serialize configuration with session/model initialization.

### Android Java API

For the complete C API and `CPUEngine`, build locally with `command/build_android.sh`. Copy `java/inspireface.jar` to your app's `libs/`, add `implementation files('libs/inspireface.jar')`, and copy `lib/<abi>/libInspireFace.so` into `app/src/main/jniLibs/<abi>/` for each ABI. The JAR automatically loads `InspireFace` on Android; all JNI entry points are built into that one library. Remove `libInspireFaceJNI.so` when upgrading from the earlier two-library build. Use the same `com.insightface.sdk.inspireface.jni` API shown above and the JAR from the matching build. The low-level binding does not require the separate Android SDK dependency below.

For builds with R8/ProGuard enabled, copy `java/consumer-rules.pro` into the app module and add it to `proguardFiles` so JNI class and field names are preserved.

Existing apps can continue using the legacy Android API with the rebuilt core library. The supplementary capture/snapshot sources remain in `java/com/insightface/sdk/inspireface/`. When combining both APIs, package only one matching copy of `libInspireFace.so` and explicitly manage native resource ownership.

We have an [Android SDK project](https://github.com/HyperInspire/inspireface-android-sdk) that integrates pre-compiled dynamic libraries, and you can use it directly.

Precompiled library support: 
- arm64-v8a
- armeabi-v7a
- x86_64

#### a. Quick to use in Android

We released InspireFace's Android SDK on JitPack, which you can incorporate into your android projects in the following ways.

- Step 1. Add the JitPack repository to your build file add it in your root **build.gradle** at the end of repositories:

  ```groovy
  allprojects {
      repositories {
         ...
         maven { url 'https://jitpack.io' }
      }
  }
  ```

- Step 2. Add the dependency

  ```groovy
  dependencies {
      implementation 'com.github.HyperInspire:inspireface-android-sdk:v1.2.4.post1'
  }
  ```

#### b. Use the Android example project

We have prepared an  [Android SDK project](https://github.com/HyperInspire/inspireface-android-sdk). You can download library from the [Release Page](https://github.com/HyperInspire/InspireFace/releases) or compile the Android library yourself and place it in the `inspireface/libs` directory of the Android sample project. You can compile and run this project using Android Studio.

```bash
inspireface-android-sdk/inspireface/libs
├── arm64-v8a
│   └── libInspireFace.so
└── armeabi-v7a
    └── libInspireFace.so
```

You need to get the resource file from the release  [Release Page](https://github.com/HyperInspire/InspireFace/releases) and place it in the `asset/inspireface` in your android project:

```
asset/
└── inspireface/
    │── Pikachu
    │── Megatron
    │── Gundam_RK356X
    └── Gundam_RK3588
```

#### How to use the Android/Java API

We provide a Java API for Android devices, which is implemented using Java Native Interface(JNI). 

```java
// Launch InspireFace, only need to call once
boolean launchStatus = InspireFace.GlobalLaunch(this, InspireFace.PIKACHU);
if (!launchStatus) {
    Log.e(TAG, "Failed to launch InspireFace");
}

// Create a ImageStream
ImageStream stream = InspireFace.CreateImageStreamFromBitmap(img, InspireFace.CAMERA_ROTATION_0);

// Create a session
CustomParameter parameter = InspireFace.CreateCustomParameter()
                .enableRecognition(true)
                .enableFaceQuality(true)
                .enableFaceAttribute(true)
                .enableInteractionLiveness(true)
                .enableLiveness(true)
                .enableMaskDetect(true);
Session session = InspireFace.CreateSession(parameter, InspireFace.DETECT_MODE_ALWAYS_DETECT, 10, -1, -1);

// Execute face detection
MultipleFaceData multipleFaceData = InspireFace.ExecuteFaceTrack(session, stream);
if (multipleFaceData.detectedNum > 0) {
    // Get face feature
    FaceFeature feature = InspireFace.ExtractFaceFeature(session, stream, multipleFaceData.tokens[0]);
    // ....
}

// .... 

// Release resource
InspireFace.ReleaseSession(session);
InspireFace.ReleaseImageStream(stream);

// Global release
InspireFace.GlobalRelease();
```

## Test

In the project, there is a subproject called `cpp/test`. To compile it, you need to enable the `ISF_BUILD_WITH_TEST` switch, which will allow you to compile executable programs for testing.

```bash
cmake -DISF_BUILD_WITH_TEST=ON ..
```
To run the test modules in the project, first check if the resource files exist in the test_res/pack directory. If they don't exist, you can either execute **command/download_models_general.sh** to download the required files, or download the files from the [Release Page](https://github.com/HyperInspire/InspireFace/releases/tag/v1.x) and manually place them in this directory.

```bash

test_res
├── data
├── images
├── pack	<-- The model package files are here
├── save
├── valid_lfw_funneled.txt
├── video
└── video_frames

```
After compilation, you can find the executable program **Test** (**Test.exe** on Windows) in `YOUR_BUILD_FOLDER/test`. The program accepts two optional parameters:

- **test_dir**：Path to the test resource files
- **pack**：Name of the model to be tested
```bash
./Test --test_dir PATH/test_res --pack Pikachu
```
During the process of building the test program using CMake, it will involve selecting CMake parameters. For specific details, you can refer to the parameter configuration table.

**Note**: If you want to view the benchmark test report, you can click on the [link](doc/Benchmark-Remark(Updating).md).

### Quick Test

If you need to perform a quick test, you can use the script we provide. This script will automatically download the test file `test_res` and build the test program to run the test. 

*Note: If you need to enable more comprehensive tests, you can adjust the options in the script as needed.*

```bash
# If you are using Ubuntu, you can execute this.
bash ci/quick_test_linux_x86_usual.sh

# For macOS or Linux, you can execute this.
bash ci/quick_test_local.sh
```

On Windows, `.\command\build_windows.ps1` builds the SDK and runs the registered CTest suite with the prepared `test_res` resources. To rerun it, use `ctest --test-dir build/windows-x64-Release-shared --output-on-failure --no-tests=error`. Automated tests also run through the configured GitHub Actions workflows.

## Features
The following Features and technologies are currently supported.

| Feature | CPU | RKNPU<sup><br/>(RV1109/1126) | RKNPU<sup><br/>(RV1103/1106) | RKNPU<sup><br/>(RK3566/3568/3588) | ANE<sup><br/>(MacOS/iOS) | GPU<sup><br/>(TensorRT) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Face Detection | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) |
| Landmark | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) |
| Face Embeddings | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | - | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) |
| Face Comparison | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | - | - | - | - | - |
| Face Recognition | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | - | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) |
| Alignment | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | - | - | - | - | - |
| Tracking | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) |
| Mask Detection | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | - | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | - | - |
| Silent Liveness | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | - | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | - | - |
| Passive Liveness<sup>Plus</sup> | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) |
| Flash Liveness<sup>Plus</sup> | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) |
| Face Quality | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) |
| Pose Estimation | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) |
| Face Attribute | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | - | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | - | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) |
| Cooperative Liveness | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) |
| Face Emotion | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | - | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) |
| Embedding Management | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | - | - | - | - | - |

- Some models and features that do **not support** NPU or GPU will **automatically use CPU** for computation when running the program.

## Resource Package List

For different scenarios, we currently provide several Packs, each containing multiple models and configurations.

| Name | Supported Devices | Note | Last Update | Link |
| --- | --- | --- | --- | --- |
| Pikachu | CPU | Lightweight edge-side models | Jun 22, 2025 | [Download](https://github.com/HyperInspire/InspireFace/releases/download/v1.x/Pikachu) |
| Megatron | CPU, GPU | Mobile and server models | Jun 15, 2025 | [Download](https://github.com/HyperInspire/InspireFace/releases/download/v1.x/Megatron) |
| Megatron_TRT | GPU | CUDA-based server models | Jun 15, 2025 | [Download](https://github.com/HyperInspire/InspireFace/releases/download/v1.x/Megatron_TRT) |
| Gundam-RV1109 | RKNPU | Supports RK1109 and RK1126 | Jun 15, 2025 | [Download](https://github.com/HyperInspire/InspireFace/releases/download/v1.x/Gundam_RV1109) |
| Gundam-RV1106 | RKNPU | Supports RV1103 and RV1106 | Jul 6, 2025 | [Download](https://github.com/HyperInspire/InspireFace/releases/download/v1.x/Gundam_RV1106) |
| Gundam-RK356X | RKNPU | Supports RK3566 and RK3568 | Jun 15, 2025 | [Download](https://github.com/HyperInspire/InspireFace/releases/download/v1.x/Gundam_RK356X) |
| Gundam-RK3588 | RKNPU | Supports RK3588 | Jun 15, 2025 | [Download](https://github.com/HyperInspire/InspireFace/releases/download/v1.x/Gundam_RK3588) |

## Short-Term Plan

- [x] Add TensorRT backend support.
- [x] Add Add C++ style header files.
- [x] Add the RKNPU backend support for Android .
- [x] Python wheels for Windows x64 and Linux/macOS on x86_64 and ARM64.
- [x] Example app project for Android and iOS samples.
- [ ] Add the batch forward feature.
- [ ] Design a scheme that can be adapted to multiple CUDA devices.

- Continue to provide more support for Rockchip NPU2 devices:
- [ ] RK3576 Series
- [ ] RK3562 Series
- [ ] RV1103B/RV1106B
- [ ] RV1126B
- [ ] RK2118

## Acknowledgement

InspireFace is built on the following libraries:

- [MNN](https://github.com/alibaba/MNN)
- [RKNN](https://github.com/rockchip-linux/rknn-toolkit)
- [RKNN2](https://github.com/airockchip/rknn-toolkit2.git)
- [librga](https://github.com/airockchip/librga.git)
- [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page)
- [sqlite](https://www.sqlite.org/index.html)
- [sqlite-vec](https://github.com/asg017/sqlite-vec)
- [Catch2](https://github.com/catchorg/Catch2)
- [yaml-cpp](https://github.com/jbeder/yaml-cpp)
- [TensorRT](https://github.com/NVIDIA/TensorRT)
