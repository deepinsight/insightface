import os
import tarfile
import click
import tqdm

def remove_suffix(filename):
    """Remove the file suffix."""
    return os.path.splitext(filename)[0]

@click.command()
@click.argument('folder_path')
@click.argument('output_filename')
@click.option('--rm-suffix', is_flag=True, default=True, help='Remove file suffixes in the archive.')
def make_tar(folder_path, output_filename, rm_suffix):
    """
    Package the specified folder into a tar file. FOLDER_PATH is the path to the folder to be packaged,
    and OUTPUT_FILENAME is the name of the output tar file.
    """
    # Collect all files to be packed for printing their info before packaging
    files_to_pack = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, start=folder_path)
            if rm_suffix:
                # Split the file name and directory name
                dirs, filename = os.path.split(arcname)
                # Remove the file suffix
                filename = remove_suffix(filename)
                # Reassemble the directory and the processed file name
                arcname = os.path.join(dirs, filename)
            files_to_pack.append((file_path, arcname))

    # Print file information before packing
    print("Printing file information before packaging...")
    for file_path, arcname in files_to_pack:
        file_size_kb = os.path.getsize(file_path) / 1024  # Convert size to kilobytes
        print(f"File: {arcname}, Size: {file_size_kb:.2f} KB")

    # Package the files
    with tarfile.open(output_filename, "w") as tar:
        print("Packing....")
        for file_path, arcname in tqdm.tqdm(files_to_pack):
            tar.add(file_path, arcname=arcname)

if __name__ == '__main__':
    make_tar()
