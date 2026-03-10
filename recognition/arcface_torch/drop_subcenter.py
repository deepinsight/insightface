#!/usr/bin/env python3
"""
Drop non-dominant sub-centers and high-confidence noisy data.

This is the PyTorch equivalent of subcenter_arcface/drop.py for use with
the arcface_torch Sub-center ArcFace pipeline.

Workflow (from Sub-center ArcFace paper):
  1. Train Sub-center ArcFace (K=3) on potentially noisy data.
  2. **This script**: For each identity, find the dominant sub-center
     (the one most images are closest to), then drop images whose angle
     to the dominant sub-center exceeds a threshold (default: 75 degrees).
  3. Retrain standard ArcFace on the cleaned dataset.

Requirements:
  - Trained backbone: model.pt (or best_model.pt)
  - Sub-center FC weights: subcenter_fc_gpu_0.pt, subcenter_fc_gpu_1.pt, ...
    (saved by train_v2_subcenter.py at end of training)

Usage:
    python drop_subcenter.py \
        --data /datasets/merged_ms1m_glint_rec \
        --model /output/subcenter_merged_ms1m_glint_r100/model.pt \
        --fc-dir /output/subcenter_merged_ms1m_glint_r100 \
        --output /datasets/merged_ms1m_glint_rec_clean \
        --threshold 75 \
        --network r100 \
        --batch-size 512
"""

import argparse
import datetime
import glob
import logging
import numbers
import os
import sys

import mxnet as mx
import numpy as np
import sklearn.preprocessing
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add arcface_torch to path for backbone imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backbones import get_model


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)


def load_backbone(model_path, network='r100', embedding_size=512):
    """Load a trained PyTorch backbone."""
    backbone = get_model(network, dropout=0.0, fp16=False, num_features=embedding_size)
    state = torch.load(model_path, map_location='cpu')
    # Handle both raw state_dict and full checkpoint formats
    if isinstance(state, dict) and 'state_dict_backbone' in state:
        state = state['state_dict_backbone']
    # Strip DDP 'module.' prefix if present
    new_state = {}
    for k, v in state.items():
        new_state[k.replace('module.', '')] = v
    backbone.load_state_dict(new_state)
    backbone.eval()
    backbone.cuda()
    return backbone


def load_subcenter_weights(fc_dir, num_classes, num_subcenters=3, embedding_size=512):
    """
    Load sub-center FC weights from per-GPU files saved by train_v2_subcenter.py.
    
    Returns:
        W: numpy array of shape (num_classes, K, embedding_size), L2-normalized
    """
    fc_files = sorted(glob.glob(os.path.join(fc_dir, 'subcenter_fc_gpu_*.pt')))
    if len(fc_files) == 0:
        raise FileNotFoundError(
            f"No subcenter_fc_gpu_*.pt files found in {fc_dir}.\n"
            "These are saved by train_v2_subcenter.py at the end of training.\n"
            "If you trained without this feature, use recover_subcenter_fc.py to recover them."
        )

    logger.info(f"Loading sub-center FC weights from {len(fc_files)} files...")

    # Sort by class_start to ensure correct order
    parts = []
    for f in fc_files:
        data = torch.load(f, map_location='cpu')
        parts.append(data)
    parts.sort(key=lambda x: x['class_start'])

    K = parts[0]['num_subcenters']
    emb = parts[0]['embedding_size']
    logger.info(f"  K={K}, embedding_size={emb}")

    # Concatenate weights from all GPUs
    # Each part has weight of shape (num_local * K, embedding_size)
    all_weights = []
    total_classes = 0
    for p in parts:
        w = p['weight'].numpy()  # (num_local * K, emb)
        num_local = p['num_local']
        total_classes += num_local
        all_weights.append(w)
    
    W = np.concatenate(all_weights, axis=0)  # (total_classes * K, emb)
    logger.info(f"  Total weight shape: {W.shape}, total classes from weights: {total_classes}")
    
    if total_classes < num_classes:
        logger.warning(f"  Weight covers {total_classes} classes but dataset has {num_classes}")
    
    # L2-normalize
    W = sklearn.preprocessing.normalize(W)
    # Reshape to (num_classes, K, emb)
    W = W.reshape(-1, K, emb)
    logger.info(f"  Final W shape: {W.shape}")
    return W


def scan_rec_identities(rec_path):
    """
    Scan a flat RecordIO (as produced by convert_to_rec.py) and group
    image indices by their label (identity).

    Returns:
        id_to_indices: dict mapping label_id -> list of record indices
        num_images: total number of images
        num_classes: number of unique labels
    """
    path_imgrec = os.path.join(rec_path, 'train.rec')
    path_imgidx = os.path.join(rec_path, 'train.idx')
    imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')

    # Read header
    s = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(s)

    if header.flag > 0:
        # Our format: flag>0, label=[num_images+1, num_classes] (or identity range)
        max_idx = int(header.label[0])
        indices = range(1, max_idx)
    else:
        indices = list(imgrec.keys)

    logger.info(f"Scanning {len(indices)} records to group by identity...")
    id_to_indices = {}
    for idx in tqdm(indices, desc="Scanning records"):
        s = imgrec.read_idx(idx)
        header, _ = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = int(label)
        if label not in id_to_indices:
            id_to_indices[label] = []
        id_to_indices[label].append(idx)

    num_images = len(indices)
    num_classes = len(id_to_indices)
    logger.info(f"Found {num_images} images across {num_classes} identities")
    return id_to_indices, num_images, num_classes


@torch.no_grad()
def get_embeddings(backbone, imgrec, indices, batch_size=512, image_size=(112, 112)):
    """
    Compute L2-normalized embeddings for a list of record indices.

    Returns:
        embeddings: numpy array (N, emb_size)
        raw_contents: list of raw image bytes (for writing to output RecordIO)
    """
    raw_contents = []
    images = []

    for idx in indices:
        s = imgrec.read_idx(idx)
        header, img_bytes = mx.recordio.unpack(s)
        raw_contents.append(img_bytes)
        img = mx.image.imdecode(img_bytes).asnumpy()  # (H, W, 3) RGB
        # Normalize: (img / 255 - 0.5) / 0.5 = img / 127.5 - 1
        img = img.transpose((2, 0, 1)).astype(np.float32)  # (3, H, W)
        img = (img / 255.0 - 0.5) / 0.5
        images.append(img)

    embeddings = []
    for i in range(0, len(images), batch_size):
        batch = np.stack(images[i:i+batch_size])
        batch_tensor = torch.from_numpy(batch).cuda()
        emb = backbone(batch_tensor).cpu().numpy()
        embeddings.append(emb)

    embeddings = np.concatenate(embeddings, axis=0)
    embeddings = sklearn.preprocessing.normalize(embeddings)
    return embeddings, raw_contents


def main(args):
    logger.info(f"Arguments: {args}")
    image_size = (112, 112)

    # 1. Load backbone
    logger.info(f"Loading backbone from {args.model} (network={args.network})")
    backbone = load_backbone(args.model, args.network, args.embedding_size)

    # 2. Load sub-center FC weights
    W = load_subcenter_weights(
        args.fc_dir,
        num_classes=0,  # auto from weight files
        num_subcenters=args.k,
        embedding_size=args.embedding_size
    )
    num_classes_from_W = W.shape[0]
    K = W.shape[1]
    logger.info(f"Loaded sub-center weights: {num_classes_from_W} classes, K={K}")

    # 3. Scan the RecordIO dataset to group images by identity
    id_to_indices, total_images, num_classes = scan_rec_identities(args.data)

    # 4. Open the RecordIO for reading
    path_imgrec = os.path.join(args.data, 'train.rec')
    path_imgidx = os.path.join(args.data, 'train.idx')
    imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')

    # 5. Create output RecordIO
    os.makedirs(args.output, exist_ok=True)
    out_rec_path = os.path.join(args.output, 'train.rec')
    out_idx_path = os.path.join(args.output, 'train.idx')
    out_rec = mx.recordio.MXIndexedRecordIO(out_idx_path, out_rec_path, 'w')

    cos_thresh = np.cos(np.pi * args.threshold / 180.0)
    logger.info(f"Angle threshold: {args.threshold}° → cosine threshold: {cos_thresh:.6f}")

    # Sort identity labels for deterministic output
    sorted_labels = sorted(id_to_indices.keys())

    new_label = 0
    write_idx = 1  # index 0 is reserved for header
    total_kept = 0
    total_dropped = 0
    identities_dropped = 0

    da = datetime.datetime.now()
    for progress_i, original_label in enumerate(tqdm(sorted_labels, desc="Processing identities")):
        if progress_i > 0 and progress_i % 5000 == 0:
            db = datetime.datetime.now()
            rate = progress_i / (db - da).total_seconds()
            logger.info(
                f"  Processed {progress_i}/{len(sorted_labels)} identities "
                f"({rate:.1f} id/s), kept {total_kept}, dropped {total_dropped}"
            )

        indices = id_to_indices[original_label]

        if original_label >= num_classes_from_W:
            logger.warning(f"Label {original_label} exceeds FC weight size ({num_classes_from_W}), skipping")
            total_dropped += len(indices)
            identities_dropped += 1
            continue

        # Get embeddings for all images of this identity
        embeddings, raw_contents = get_embeddings(
            backbone, imgrec, indices, args.batch_size, image_size
        )

        # Get K sub-centers for this identity
        subcenters = W[original_label]  # (K, emb_size)

        # Find dominant sub-center: the one most images are closest to
        K_stat = np.zeros(K, dtype=np.int64)
        for i in range(embeddings.shape[0]):
            sim = np.dot(subcenters, embeddings[i])  # (K,)
            mc = np.argmax(sim)
            K_stat[mc] += 1
        dominant_index = np.argmax(K_stat)
        dominant_center = subcenters[dominant_index]

        # Compute cosine similarity of each image to the dominant sub-center
        sim = np.dot(embeddings, dominant_center)  # (N,)
        keep_mask = sim > cos_thresh

        kept_indices = np.where(keep_mask)[0]
        num_drop = embeddings.shape[0] - len(kept_indices)
        total_dropped += num_drop

        if len(kept_indices) == 0:
            # Entire identity is dropped
            identities_dropped += 1
            continue

        total_kept += len(kept_indices)

        # Write kept images with new sequential label
        for ki in kept_indices:
            header = mx.recordio.IRHeader(flag=0, label=float(new_label), id=0, id2=0)
            s = mx.recordio.pack(header, raw_contents[ki])
            out_rec.write_idx(write_idx, s)
            write_idx += 1

        new_label += 1

    # Write header at index 0
    header0 = mx.recordio.IRHeader(flag=1, label=[write_idx, new_label], id=0, id2=0)
    out_rec.write_idx(0, mx.recordio.pack(header0, b''))
    out_rec.close()

    # Write property file
    with open(os.path.join(args.output, 'property'), 'w') as f:
        f.write(f"{new_label},{image_size[0]},{image_size[1]}\n")

    # Copy validation .bin files to the output directory
    for name in ['lfw', 'cfp_fp', 'agedb_30']:
        src = os.path.join(args.data, f'{name}.bin')
        dst = os.path.join(args.output, f'{name}.bin')
        if os.path.exists(src) and not os.path.exists(dst):
            import shutil
            shutil.copy2(src, dst)
            logger.info(f"Copied validation file: {name}.bin")

    elapsed = (datetime.datetime.now() - da).total_seconds()
    logger.info("=" * 60)
    logger.info(f"Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    logger.info(f"Original:  {total_images} images, {len(sorted_labels)} identities")
    logger.info(f"Kept:      {total_kept} images, {new_label} identities")
    logger.info(f"Dropped:   {total_dropped} images, {identities_dropped} identities entirely removed")
    logger.info(f"Output:    {args.output}")
    logger.info("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Drop non-dominant sub-centers and noisy data (PyTorch version)')
    parser.add_argument('--data', required=True,
                        help='Path to input RecordIO dataset directory (containing train.rec)')
    parser.add_argument('--model', required=True,
                        help='Path to trained backbone .pt file')
    parser.add_argument('--fc-dir', required=True,
                        help='Directory containing subcenter_fc_gpu_*.pt files')
    parser.add_argument('--output', required=True,
                        help='Path to output cleaned RecordIO dataset directory')
    parser.add_argument('--network', default='r100',
                        help='Backbone network name (default: r100)')
    parser.add_argument('--embedding-size', default=512, type=int,
                        help='Embedding dimension (default: 512)')
    parser.add_argument('--batch-size', default=512, type=int,
                        help='Batch size for inference (default: 512)')
    parser.add_argument('--threshold', default=75.0, type=float,
                        help='Angle threshold in degrees (default: 75)')
    parser.add_argument('--k', default=3, type=int,
                        help='Number of sub-centers K (default: 3)')
    main(parser.parse_args())
