import os
import cv2
import mxnet as mx
from tqdm import tqdm

dataset_dir = "/datasets/ms1m-retinaface-t1"
rec_path = os.path.join(dataset_dir, "train.rec")
idx_path = os.path.join(dataset_dir, "train.idx")
out_dir = os.path.join(dataset_dir, "ms1m-retinaface-t1-images")

os.makedirs(out_dir, exist_ok=True)

# Open RecordIO
imgrec = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'r')

# Read index 0 header (usually metadata)
s = imgrec.read_idx(0)
header, _ = mx.recordio.unpack(s)

# In many face datasets:
# - header.label[0] = total images (or start idx)
# - header.label[1] = total identities (or end idx)
# But this varies by packing script, so we just scan indices safely.

# Try to use keys from idx file
keys = list(imgrec.keys)

print(f"Total record keys found: {len(keys)}")

for i in tqdm(keys):
    if i == 0:
        continue  # metadata record in many datasets
    try:
        s = imgrec.read_idx(i)
        if s is None:
            continue
        header, img = mx.recordio.unpack(s)

        # decode image bytes
        img_np = mx.image.imdecode(img).asnumpy()

        # label can be float, list, tuple, or numpy array
        label = header.label
        import numbers
        if isinstance(label, numbers.Number):
            person_id = int(label)
        else:
            person_id = int(label[0])

        person_dir = os.path.join(out_dir, f"{person_id:07d}")
        os.makedirs(person_dir, exist_ok=True)

        img_path = os.path.join(person_dir, f"{i:08d}.jpg")
        cv2.imwrite(img_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    except Exception as e:
        if "empty buffer" not in str(e):
            print(f"Failed at index {i}: {e}")