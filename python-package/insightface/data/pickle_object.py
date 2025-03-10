import sys
import os
import os.path as osp
from pathlib import Path
import pickle

def get_object(name):
    if getattr(sys, 'frozen', False):
        base_dir = sys._MEIPASS
    else:
        base_dir = Path(__file__).parent.absolute()

    objects_dir = osp.join(base_dir, 'objects')

    if not name.endswith('.pkl'):
        name = name + ".pkl"

    filepath = osp.join(objects_dir, name)
    
    if not osp.exists(filepath):
        print(f"[Error] File not found: {filepath}")
        return None

    with open(filepath, 'rb') as f:
        obj = pickle.load(f)

    return obj
