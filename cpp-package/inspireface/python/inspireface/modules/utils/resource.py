import os
import sys
from pathlib import Path
import urllib.request
import ssl
import hashlib

def get_file_hash_sha256(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

class ResourceManager:
    def __init__(self):
        """Initialize resource manager and create necessary directories"""
        self.user_home = Path.home()
        self.base_dir = self.user_home / '.inspireface'
        self.models_dir = self.base_dir / 'models'
        
        # Create directories
        self.base_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        # Model URLs
        self._MODEL_LIST = {
            "Pikachu": {
                "url": "https://github.com/HyperInspire/InspireFace/releases/download/v1.x/Pikachu",
                "filename": "Pikachu",
                "md5": "f2983a2d884902229c1443fdc921b8e5f49cf2daba8a4f103cd127910dc9e7cd"
            },
            "Megatron": {
                "url": "https://github.com/HyperInspire/InspireFace/releases/download/v1.x/Megatron",
                "filename": "Megatron",
                "md5": "28f2284c5e7cf53b0e152ff524a416c966ab21e724002643b1304aedc4af6b06"
            }
        }

    def get_model(self, name: str, re_download: bool = False) -> str:
        """
        Get model path. Download if not exists or re_download is True.
        
        Args:
            name: Model name
            re_download: Force re-download if True
            
        Returns:
            str: Full path to model file
        """
        if name not in self._MODEL_LIST:
            raise ValueError(f"Model '{name}' not found. Available models: {list(self._MODEL_LIST.keys())}")

        model_info = self._MODEL_LIST[name]
        model_file = self.models_dir / model_info["filename"]
        downloading_flag = model_file.with_suffix('.downloading')
        
        # Check if model exists and is complete
        if model_file.exists() and not downloading_flag.exists() and not re_download:
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
        
# Usage example
if __name__ == "__main__":
    try:
        rm = ResourceManager()
        model_path = rm.get_model("Pikachu")
        print(f"Model path: {model_path}")
        
    except Exception as e:
        print(f"Error: {e}")