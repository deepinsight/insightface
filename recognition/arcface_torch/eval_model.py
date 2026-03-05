"""
Evaluate a face recognition model on lfw.bin, cfp_fp.bin, agedb_30.bin
using the same verification protocol as the training pipeline.

Supports both ONNX (.onnx) and PyTorch (.pt) checkpoint formats.

Usage (ONNX):
    python eval_model.py \
        --model /output/ms1mv3_r100/model.onnx \
        --bin-dir /datasets/merged_ms1m_glint_rec \
        --targets lfw cfp_fp agedb_30

Usage (PyTorch checkpoint):
    python eval_model.py \
        --model /output/merged_ms1m_glint_r100_2nd_try/model.pt \
        --network r100 \
        --bin-dir /datasets/merged_ms1m_glint_rec \
        --targets lfw cfp_fp agedb_30

Usage (best_model.pt from validation callback):
    python eval_model.py \
        --model /output/merged_ms1m_glint_r100_2nd_try/best_model.pt \
        --network r100 \
        --bin-dir /datasets/merged_ms1m_glint_rec
"""
import argparse
import os
import sys
import numpy as np
import torch

# Reuse the existing verification code and backbones
sys.path.insert(0, os.path.dirname(__file__))
from eval import verification
from backbones import get_model


class ONNXBackbone:
    """Wraps an ONNX model to behave like a PyTorch backbone for verification.test()"""

    def __init__(self, onnx_path, device='cuda'):
        import onnxruntime as ort
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        output_shape = self.session.get_outputs()[0].shape
        print(f"ONNX model loaded: input={input_shape}, output={output_shape}")
        print(f"Providers: {self.session.get_providers()}")

    def __call__(self, x):
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


def load_pytorch_backbone(model_path, network='r100', embedding_size=512, device='cuda'):
    """Load a PyTorch backbone from a state_dict .pt file."""
    backbone = get_model(network, fp16=False, num_features=embedding_size)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)

    # Handle different checkpoint formats
    if 'state_dict_backbone' in state_dict:
        # Full checkpoint (from save_all_states=True)
        state_dict = state_dict['state_dict_backbone']
        print(f"Loaded backbone from full checkpoint")
    
    # Remove 'module.' prefix if saved from DDP
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v

    backbone.load_state_dict(new_state_dict)
    backbone = backbone.to(device)
    backbone.eval()
    print(f"PyTorch model loaded: network={network}, embedding_size={embedding_size}")
    print(f"Parameters: {sum(p.numel() for p in backbone.parameters()):,}")
    return CUDAWrapper(backbone, device)


class CUDAWrapper(torch.nn.Module):
    """Wraps a backbone to auto-move input tensors to the model's device."""
    def __init__(self, backbone, device):
        super().__init__()
        self.backbone = backbone
        self.device = device

    def forward(self, x):
        return self.backbone(x.to(self.device))


def main():
    parser = argparse.ArgumentParser(description='Evaluate face recognition model on verification benchmarks')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model (.onnx or .pt)')
    parser.add_argument('--network', type=str, default='r100',
                        help='Backbone architecture for .pt models (default: r100)')
    parser.add_argument('--embedding-size', type=int, default=512,
                        help='Embedding dimension (default: 512)')
    parser.add_argument('--bin-dir', type=str, required=True,
                        help='Directory containing .bin validation files')
    parser.add_argument('--targets', nargs='+', default=['lfw', 'cfp_fp', 'agedb_30'],
                        help='Validation targets (default: lfw cfp_fp agedb_30)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for inference')
    parser.add_argument('--nfolds', type=int, default=10,
                        help='Number of folds for evaluation')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()

    # Auto-detect model format
    ext = os.path.splitext(args.model)[1].lower()
    if ext == '.onnx':
        backbone = ONNXBackbone(args.model, device=args.device)
    elif ext == '.pt':
        backbone = load_pytorch_backbone(
            args.model, network=args.network,
            embedding_size=args.embedding_size, device=args.device)
    else:
        raise ValueError(f"Unsupported model format: {ext}. Use .onnx or .pt")

    print(f"\n{'='*60}")
    print(f"Evaluating: {args.model}")
    print(f"Benchmarks: {args.targets}")
    print(f"{'='*60}")

    results = {}
    for target in args.targets:
        bin_path = os.path.join(args.bin_dir, f"{target}.bin")
        if not os.path.exists(bin_path):
            print(f"\n[SKIP] {target}.bin not found at {bin_path}")
            continue

        print(f"\n--- {target} ---")
        data_set = verification.load_bin(bin_path, image_size=(112, 112))
        acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(
            data_set, backbone, args.batch_size, nfolds=args.nfolds)

        print(f"[{target}] XNorm:         {xnorm:.4f}")
        print(f"[{target}] Accuracy-Flip: {acc2:.5f} +/- {std2:.5f}")
        results[target] = {'acc': acc2, 'std': std2, 'xnorm': xnorm}

    # Summary table
    print(f"\n{'='*60}")
    print(f"{'Benchmark':<12} {'Accuracy':>10} {'Std':>10} {'XNorm':>10}")
    print(f"{'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    for name, r in results.items():
        print(f"{name:<12} {r['acc']:>10.5f} {r['std']:>10.5f} {r['xnorm']:>10.4f}")
    if results:
        mean_acc = np.mean([r['acc'] for r in results.values()])
        print(f"{'-'*12} {'-'*10} {'-'*10} {'-'*10}")
        print(f"{'Mean':<12} {mean_acc:>10.5f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
