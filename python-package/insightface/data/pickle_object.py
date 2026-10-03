import sys
import os
import os.path as osp
from pathlib import Path
import pickle

def get_object(name):
    base_dirs = []
    if getattr(sys, 'frozen', False):
        base_dirs.append(sys._MEIPASS)
    base_dirs.append(Path(__file__).parent.absolute())

    if not name.endswith('.pkl'):
        name = name + ".pkl"

    for base_dir in base_dirs:
        filepath = osp.join(base_dir, 'objects', name)
        if osp.exists(filepath):
            with open(filepath, 'rb') as f:
                return pickle.load(f)

    print(f"[Error] File not found: {filepath}")
    return None
