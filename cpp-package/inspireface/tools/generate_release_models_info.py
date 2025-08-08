import hashlib
import os
import json
import click

need_models = [
    "Pikachu",
    "Megatron",
    "Megatron_TRT",
    "Gundam_RK356X",
    "Gundam_RK3588",
]

def get_file_hash_sha256(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

@click.command()
@click.argument('model_dir', default="test_res/pack/")
@click.option('--models', '-m', multiple=True, help='Specify the model name to process')
def main(model_dir, models):
    model_info = {}
    
    # If no model is specified, process all predefined models
    models_to_process = models if models else need_models
    
    for file in models_to_process:
        file_path = os.path.join(model_dir, file)
        if os.path.exists(file_path):
            md5 = get_file_hash_sha256(file_path)
            model_info[file] = {
                "url": f"https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/inspireface_modelzoo/t4/{file}",
                "filename": file,
                "md5": md5
            }
        else:
            print(f"Warning: File {file_path} does not exist")
    
    print("\033[33mNeed to modify the python/inspireface/modules/utils/resource.py, changes to the information release\033[0m")
    # Output the result
    print(json.dumps(model_info, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    main()


