"""
Evaluate an ONNX face recognition model on lfw.bin, cfp_fp.bin, agedb_30.bin
using the same verification protocol as the training pipeline.

Usage:
    python eval_onnx.py \
        --model /path/to/model.onnx \
        --bin-dir /path/to/dir/with/lfw.bin \
        --targets lfw cfp_fp agedb_30
"""
import argparse
import os
import sys
import numpy as np
import torch
import onnxruntime as ort

# Reuse the existing verification code
sys.path.insert(0, os.path.dirname(__file__))
from eval import verification


class ONNXBackbone:
    """Wraps an ONNX model to behave like a PyTorch backbone for verification.test()"""

    def __init__(self, onnx_path, device='cuda'):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        output_shape = self.session.get_outputs()[0].shape
        print(f"ONNX model loaded: input={input_shape}, output={output_shape}")
        print(f"Providers: {self.session.get_providers()}")

    def __call__(self, x):
        """x is a torch.Tensor [B, 3, 112, 112]. Returns a torch.Tensor of embeddings."""
        if isinstance(x, torch.Tensor):
            x_np = x.numpy() if x.device.type == 'cpu' else x.cpu().numpy()
        else:
            x_np = x
        x_np = x_np.astype(np.float32)
        outputs = self.session.run(None, {self.input_name: x_np})
        return torch.from_numpy(outputs[0])

    def eval(self):
        return self

    def train(self, mode=True):
        return self


def main():
    parser = argparse.ArgumentParser(description='Evaluate ONNX model on face verification benchmarks')
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--bin-dir', type=str, required=True, help='Directory containing .bin validation files')
    parser.add_argument('--targets', nargs='+', default=['lfw', 'cfp_fp', 'agedb_30'],
                        help='Validation targets (default: lfw cfp_fp agedb_30)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for inference')
    parser.add_argument('--nfolds', type=int, default=10, help='Number of folds for evaluation')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()

    backbone = ONNXBackbone(args.model, device=args.device)

    print(f"\n{'='*60}")
    print(f"Evaluating: {args.model}")
    print(f"Benchmarks: {args.targets}")
    print(f"{'='*60}\n")

    for target in args.targets:
        bin_path = os.path.join(args.bin_dir, f"{target}.bin")
        if not os.path.exists(bin_path):
            print(f"[SKIP] {target}.bin not found at {bin_path}")
            continue

        print(f"\n--- {target} ---")
        data_set = verification.load_bin(bin_path, image_size=(112, 112))
        acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(
            data_set, backbone, args.batch_size, nfolds=args.nfolds)

        print(f"[{target}] XNorm:         {xnorm:.4f}")
        print(f"[{target}] Accuracy-Flip: {acc2:.5f} +/- {std2:.5f}")

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == '__main__':
    main()
