"""
InspireFace Resource Manager

This module provides model downloading functionality with two modes:
1. Original mode: Download models from COS (Tencent Cloud Object Storage)
2. ModelScope mode: Download models from ModelScope platform

ModelScope mode usage:
    rm = ResourceManager(use_modelscope=True, modelscope_model_id="tunmxy/InspireFace")
    model_path = rm.get_model("Gundam_RV1106")

Requirements for ModelScope mode:
    pip install modelscope
"""

import os
import sys
from pathlib import Path
import urllib.request
import ssl
import hashlib

try:
    from modelscope.hub.snapshot_download import snapshot_download
    MODELSCOPE_AVAILABLE = True
except ImportError:
    MODELSCOPE_AVAILABLE = False

# Global configuration for resource downloading
USE_OSS_DOWNLOAD = False  # If True, force use OSS download instead of ModelScope

def set_use_oss_download(use_oss: bool):
    """Set whether to use OSS download instead of ModelScope
    
    Args:
        use_oss (bool): If True, use OSS download; if False, use ModelScope (default)
    """
    global USE_OSS_DOWNLOAD
    USE_OSS_DOWNLOAD = use_oss

def get_file_hash_sha256(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

class ResourceManager:
    def __init__(self, use_modelscope: bool = True, modelscope_model_id: str = "tunmxy/InspireFace"):
        """Initialize resource manager and create necessary directories
        
        Args:
            use_modelscope: Whether to download models from ModelScope platform
            modelscope_model_id: ModelScope model ID (default: tunmxy/InspireFace)
        """
        self.user_home = Path.home()
        self.base_dir = self.user_home / '.inspireface'
        self.models_dir = self.base_dir / 'models'
        
        # ModelScope configuration
        self.use_modelscope = use_modelscope
        self.modelscope_model_id = modelscope_model_id
        self.modelscope_cache_dir = self.base_dir / 'ms'
        
        # Create directories
        self.base_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        if self.use_modelscope:
            self.modelscope_cache_dir.mkdir(exist_ok=True)
            
        # Check ModelScope availability
        if self.use_modelscope and not MODELSCOPE_AVAILABLE:
            raise ImportError(
                "ModelScope is not available. You have two options:\n"
                "1. Install ModelScope: pip install modelscope\n" 
                "2. Switch to OSS download mode by calling: inspireface.use_oss_download(True) before using InspireFace"
            )
        
        # Model URLs
        self._MODEL_LIST = {
            "Pikachu": {
                "url": "https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/inspireface_modelzoo/t4/Pikachu",
                "filename": "Pikachu",
                "md5": "5037ba1f49905b783a1c973d5d58b834a645922cc2814c8e3ca630a38dc24431"
            },
            "Megatron": {
                "url": "https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/inspireface_modelzoo/t4/Megatron",
                "filename": "Megatron",
                "md5": "709fddf024d9f34ec034d8ef79a4779e1543b867b05e428c1d4b766f69287050"
            },
            "Megatron_TRT": {
                "url": "https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/inspireface_modelzoo/t4/Megatron_TRT",
                "filename": "Megatron_TRT",
                "md5": "bc9123bdc510954b28d703b8ffe6023f469fb81123fd0b0b27fd452dfa369bab"
            },
            "Gundam_RK356X": {
                "url": "https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/inspireface_modelzoo/t4/Gundam_RK356X",
                "filename": "Gundam_RK356X",
                "md5": "0fa12a425337ed98bd82610768a50de71cf93ef42a0929ba06cc94c86f4bd415"
            },
            "Gundam_RK3588": {
                "url": "https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/inspireface_modelzoo/t4/Gundam_RK3588",
                "filename": "Gundam_RK3588",
                "md5": "66070e8d654408b666a2210bd498a976bbad8b33aef138c623e652f8d956641e"
            }
        }

    def _download_from_modelscope(self, model_name: str) -> str:
        """Download model from ModelScope platform
        
        Args:
            model_name: Name of the model to download
            
        Returns:
            str: Path to the downloaded model file
        """
        if not MODELSCOPE_AVAILABLE:
            raise ImportError("ModelScope is not available. Please install it with: pip install modelscope")
            
        print(f"Downloading model '{model_name}' from ModelScope...")
        
        try:
            # Download specific model file from ModelScope
            cache_dir = snapshot_download(
                model_id=self.modelscope_model_id,
                cache_dir=str(self.modelscope_cache_dir),
                allow_file_pattern=[model_name]  # Only download the specific model file
            )
            
            model_file_path = Path(cache_dir) / model_name
            
            if not model_file_path.exists():
                raise FileNotFoundError(f"Model file '{model_name}' not found in downloaded repository")
                
            print(f"ModelScope download completed: {model_file_path}")
            return str(model_file_path)
            
        except Exception as e:
            raise RuntimeError(f"Failed to download model from ModelScope: {e}")

    def get_model(self, name: str, re_download: bool = False, ignore_verification: bool = False) -> str:
        """
        Get model path. Download if not exists or re_download is True.
        
        Args:
            name: Model name
            re_download: Force re-download if True
            ignore_verification: Skip model hash verification if True
            
        Returns:
            str: Full path to model file
        """
        # Check global OSS setting first, then instance setting
        use_oss = USE_OSS_DOWNLOAD
        use_modelscope_actual = self.use_modelscope and not use_oss
        
        # Use ModelScope download if enabled and OSS is not forced
        if use_modelscope_actual:
            return self._download_from_modelscope_with_cache(name, re_download)
            
        # Original download logic for backwards compatibility
        if name not in self._MODEL_LIST:
            raise ValueError(f"Model '{name}' not found. Available models: {list(self._MODEL_LIST.keys())}")

        model_info = self._MODEL_LIST[name]
        model_file = self.models_dir / model_info["filename"]
        downloading_flag = model_file.with_suffix('.downloading')
        
        # Check if model exists and is complete
        if model_file.exists() and not downloading_flag.exists() and not re_download:
            if ignore_verification:
                print(f"Warning: Model verification skipped for '{name}' as requested.")
                return str(model_file)
                
            current_hash = get_file_hash_sha256(model_file)
            if current_hash == model_info["md5"]:
                return str(model_file)
            else:
                print(f"Model file hash mismatch for '{name}'. Re-downloading...")

        # Start download
        try:
            print(f"Downloading model '{name}'...")
            downloading_flag.touch()

            # Create SSL context and headers
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            
            req = urllib.request.Request(model_info["url"], headers=headers)
            
            with urllib.request.urlopen(req, context=ssl_context) as response:
                total_size = int(response.headers.get('content-length', 0))
                block_size = 8192
                downloaded_size = 0
                
                with open(model_file, 'wb') as f:
                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        downloaded_size += len(buffer)
                        f.write(buffer)
                        
                        if total_size > 0:
                            percent = (downloaded_size / total_size) * 100
                            sys.stdout.write(f"\rDownloading {name}: {percent:.1f}%")
                            sys.stdout.flush()
            
            print("\nDownload completed")
            downloading_flag.unlink()  # Remove the downloading flag
            return str(model_file)

        except Exception as e:
            if model_file.exists():
                model_file.unlink()
            if downloading_flag.exists():
                downloading_flag.unlink()
            raise RuntimeError(f"Failed to download model: {e}")

    def _download_from_modelscope_with_cache(self, name: str, re_download: bool = False) -> str:
        """Download model from ModelScope with local caching logic
        
        Args:
            name: Model name
            re_download: Force re-download if True
            
        Returns:
            str: Path to the model file
        """
        # Check if model exists in ModelScope cache
        model_file_path = self.modelscope_cache_dir / name
        
        if model_file_path.exists() and not re_download:
            print(f"Using cached model '{name}' from ModelScope")
            return str(model_file_path)
            
        # Download from ModelScope
        return self._download_from_modelscope(name)
        
# Usage examples
if __name__ == "__main__":
    try:
        # Example 1: Default mode (ModelScope)
        print("=== Default mode (ModelScope) ===")
        rm = ResourceManager()
        model_path = rm.get_model("Gundam_RV1106")
        print(f"ModelScope model path: {model_path}")
        
        # Example 2: Force OSS mode using global setting
        print("\n=== OSS mode (global setting) ===")
        set_use_oss_download(True)
        rm_oss = ResourceManager()
        model_path_oss = rm_oss.get_model("Pikachu")
        print(f"OSS model path: {model_path_oss}")
        
        # Reset to default
        set_use_oss_download(False)
        
        # Example 3: Explicit ModelScope mode
        print("\n=== Explicit ModelScope mode ===")
        rm_ms = ResourceManager(use_modelscope=True, modelscope_model_id="tunmxy/InspireFace")
        model_path_ms = rm_ms.get_model("Gundam_RV1106")
        print(f"Explicit ModelScope model path: {model_path_ms}")
        
    except Exception as e:
        print(f"Error: {e}")