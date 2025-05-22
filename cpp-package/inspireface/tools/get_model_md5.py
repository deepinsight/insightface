import hashlib
import os
import click

file_list = [
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
def main(model_dir):
    for file in file_list:
        print(f"{file}: {get_file_hash_sha256(os.path.join(model_dir, file))}")

if __name__ == "__main__":
    main()
