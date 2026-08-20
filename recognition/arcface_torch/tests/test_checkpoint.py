"""utils/utils_checkpoint.py: resumable checkpoints shared by all train_*.py scripts."""
import os
import sys
import tempfile
sys.path.insert(0, ".")
import pytest
import torch
from easydict import EasyDict as edict


def _state(seed):
    torch.manual_seed(seed)
    backbone = torch.nn.Linear(4, 3)
    pfc = torch.nn.Linear(3, 5)
    opt = torch.optim.SGD(list(backbone.parameters()) + list(pfc.parameters()), lr=0.1, momentum=0.9)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    return backbone, pfc, opt, sched, scaler


def _cfg(tmp, **kw):
    cfg = edict(output=tmp, resume=False, resume_from=None, resume_epoch=None,
                save_all_states=True, keep_epoch_checkpoints=True, num_epoch=5)
    cfg.update(kw)
    return cfg


def test_save_writes_latest_and_per_epoch_files_with_scaler_state():
    from utils.utils_checkpoint import save_checkpoint
    with tempfile.TemporaryDirectory() as tmp:
        b, p, o, s, sc = _state(0)
        cfg = _cfg(tmp)
        save_checkpoint(cfg, rank=0, epoch=2, global_step=300, backbone=b, module_partial_fc=p,
                        opt=o, lr_scheduler=s, grad_scaler=sc)
        assert os.path.exists(os.path.join(tmp, "checkpoint_gpu_0.pt"))
        assert os.path.exists(os.path.join(tmp, "checkpoint_epoch2_gpu_0.pt"))
        ck = torch.load(os.path.join(tmp, "checkpoint_gpu_0.pt"), map_location="cpu", weights_only=False)
        assert ck["epoch"] == 3 and ck["global_step"] == 300 and ck["num_epoch"] == 5
        for k in ("state_dict_backbone", "state_dict_softmax_fc", "state_optimizer",
                  "state_lr_scheduler", "state_grad_scaler"):
            assert k in ck, k


def test_save_without_keep_epoch_checkpoints_only_writes_latest():
    from utils.utils_checkpoint import save_checkpoint
    with tempfile.TemporaryDirectory() as tmp:
        b, p, o, s, sc = _state(0)
        save_checkpoint(_cfg(tmp, keep_epoch_checkpoints=False), 0, 1, 10, b, p, o, s, sc)
        assert os.listdir(tmp) == ["checkpoint_gpu_0.pt"]


def test_resolve_resume_path_variants():
    from utils.utils_checkpoint import resolve_resume_path
    with tempfile.TemporaryDirectory() as tmp:
        assert resolve_resume_path(_cfg(tmp, resume=True), rank=1) == os.path.join(tmp, "checkpoint_gpu_1.pt")
        assert resolve_resume_path(_cfg(tmp, resume=True, resume_epoch=3), rank=0) == \
            os.path.join(tmp, "checkpoint_epoch3_gpu_0.pt")
        other = os.path.join(tmp, "other_run")
        os.makedirs(other)
        assert resolve_resume_path(_cfg(tmp, resume=True, resume_from=other), rank=0) == \
            os.path.join(other, "checkpoint_gpu_0.pt")
        assert resolve_resume_path(_cfg(tmp, resume=True, resume_from=other, resume_epoch=4), rank=1) == \
            os.path.join(other, "checkpoint_epoch4_gpu_1.pt")
        pat = os.path.join(other, "ck_{rank}.pt")
        assert resolve_resume_path(_cfg(tmp, resume=True, resume_from=pat), rank=2) == os.path.join(other, "ck_2.pt")
        assert resolve_resume_path(_cfg(tmp, resume=False), rank=0) is None


def test_load_restores_all_states_and_returns_position():
    from utils.utils_checkpoint import save_checkpoint, load_checkpoint
    with tempfile.TemporaryDirectory() as tmp:
        b, p, o, s, sc = _state(0)
        for _ in range(3):                                         # advance optimizer / scheduler state
            o.zero_grad(); (p(b(torch.randn(2, 4))) ** 2).sum().backward(); o.step(); s.step()
        save_checkpoint(_cfg(tmp), 0, 2, 300, b, p, o, s, sc)
        b2, p2, o2, s2, sc2 = _state(1)
        start_epoch, global_step = load_checkpoint(_cfg(tmp, resume=True), 0, b2, p2, o2, s2, sc2)
        assert (start_epoch, global_step) == (3, 300)
        assert torch.equal(b2.weight, b.weight) and torch.equal(p2.weight, p.weight)
        assert s2.last_epoch == s.last_epoch and s2.get_last_lr() == s.get_last_lr()
        mom = o.state[b.weight]["momentum_buffer"]; mom2 = o2.state[b2.weight]["momentum_buffer"]
        assert torch.equal(mom, mom2)


def test_load_specific_epoch_and_tolerates_old_checkpoints_without_scaler():
    from utils.utils_checkpoint import save_checkpoint, load_checkpoint
    with tempfile.TemporaryDirectory() as tmp:
        b, p, o, s, sc = _state(0)
        save_checkpoint(_cfg(tmp), 0, 0, 100, b, p, o, s, sc)
        save_checkpoint(_cfg(tmp), 0, 1, 200, b, p, o, s, sc)
        # strip the scaler entry from epoch-1 file to emulate a checkpoint from the old scripts
        path = os.path.join(tmp, "checkpoint_epoch1_gpu_0.pt")
        ck = torch.load(path, map_location="cpu", weights_only=False); del ck["state_grad_scaler"]; torch.save(ck, path)
        b2, p2, o2, s2, sc2 = _state(1)
        assert load_checkpoint(_cfg(tmp, resume=True, resume_epoch=0), 0, b2, p2, o2, s2, sc2) == (1, 100)
        assert load_checkpoint(_cfg(tmp, resume=True, resume_epoch=1), 0, b2, p2, o2, s2, sc2) == (2, 200)


def test_load_warns_when_num_epoch_differs(caplog=None):
    """The LR schedule was built for checkpoint['num_epoch']; resuming with another value is allowed
    but must be visible in the log (the restored scheduler keeps the old total_iters)."""
    import logging
    from utils.utils_checkpoint import save_checkpoint, load_checkpoint
    with tempfile.TemporaryDirectory() as tmp:
        b, p, o, s, sc = _state(0)
        save_checkpoint(_cfg(tmp, num_epoch=5), 0, 1, 200, b, p, o, s, sc)
        records = []
        h = logging.Handler(); h.emit = lambda r: records.append(r.getMessage())
        logging.getLogger().addHandler(h); logging.getLogger().setLevel(logging.INFO)
        try:
            load_checkpoint(_cfg(tmp, resume=True, num_epoch=8), 0, *_state(1))
        finally:
            logging.getLogger().removeHandler(h)
        assert any("num_epoch" in m and "5" in m and "8" in m for m in records), records


def test_load_missing_file_raises_clear_error():
    from utils.utils_checkpoint import load_checkpoint
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(FileNotFoundError):
            load_checkpoint(_cfg(tmp, resume=True), 0, *_state(0))


def test_load_pretrained_backbone_initialises_from_model_pt():
    """`config.pretrained` points at a model.pt (backbone state_dict, possibly saved from DDP with a
    'module.' prefix); it must initialise matching tensors, tolerate head/shape mismatches and report them."""
    from utils.utils_checkpoint import load_pretrained_backbone
    with tempfile.TemporaryDirectory() as tmp:
        torch.manual_seed(0)
        src = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 5))
        path = os.path.join(tmp, "model.pt")
        torch.save({"module." + k: v for k, v in src.state_dict().items()}, path)
        torch.manual_seed(1)
        dst = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 7))   # last layer differs
        report = load_pretrained_backbone(dst, path)
        assert torch.equal(dst[0].weight, src[0].weight) and torch.equal(dst[0].bias, src[0].bias)
        assert not torch.equal(dst[1].weight[:5], src[1].weight)
        assert report["loaded"] == 2 and set(report["skipped"]) == {"1.weight", "1.bias"}
        wrapped = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 5))
        wrapped.module = None                        # DDP-like object with a .module attribute is unwrapped
        ddp_like = type("DDP", (), {"module": wrapped})()
        assert load_pretrained_backbone(ddp_like, path)["loaded"] == 4
        assert torch.equal(wrapped[1].weight, src[1].weight)
