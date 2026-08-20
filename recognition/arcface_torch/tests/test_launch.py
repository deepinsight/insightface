"""launch.py: one-command interactive launcher (dataset inspection, setting recommendation, config rendering)."""
import os
import sys
sys.path.insert(0, ".")
import pytest

MOCK = "/data/fu.dai/face_dataset/webface260m_mock_100mb"


@pytest.mark.skipif(not os.path.isdir(MOCK), reason="mock dataset not available")
def test_inspect_dataset_reads_counts_from_recordio():
    from launch import inspect_dataset
    info = inspect_dataset(MOCK)
    assert info["num_image"] == 37173 and info["num_classes"] == 544
    assert info["rec_bytes"] > 100_000_000


def test_recommend_webface42m_on_4xh200_cnn():
    from launch import recommend
    r = recommend("r100", "arcface", num_gpus=4, gpu_mem_gb=141, num_classes=2_059_906, num_image=42_474_557)
    assert r["script"] == "train_v2.py" and r["optimizer"] == "sgd"
    assert r["sample_rate"] == 0.3                       # PartialFC for >1M classes (official WebFace42M configs)
    assert r["batch_size"] == 512 and r["total_batch"] == 2048
    assert abs(r["lr"] - 0.2) < 1e-9                     # 0.1 @ 1024 total, scaled linearly
    assert r["fp16"] is True and r["num_epoch"] == 20 and r["margin_list"] == (1.0, 0.5, 0.0)


def test_recommend_vit_and_transface_pp_use_adamw_and_right_scripts():
    from launch import recommend
    r = recommend("vit_b_dp005_mask_005", "arcface", 4, 141, 2_059_906, 42_474_557)
    assert r["script"] == "train_v2.py" and r["optimizer"] == "adamw" and r["batch_size"] == 256
    assert abs(r["lr"] - 0.0005) < 1e-12                  # 1e-3 @ 2048 total -> 1024 total
    r = recommend("transface_pp_vit_s", "adaface", 4, 141, 2_059_906, 42_474_557)
    assert r["script"] == "train_transface_pp.py" and r["loss"] == "adaface" and r["byte_format"] == "tiff"
    r = recommend("transface_vit_b", "arcface", 4, 141, 2_059_906, 42_474_557)
    assert r["script"] == "train_transface.py" and r["dpap_prob"] == 0.2 and r["ehsm"] is True
    r = recommend("r50", "adaface", 4, 141, 2_059_906, 42_474_557)
    assert r["script"] == "train_adaface.py"


def test_recommend_scales_with_gpu_memory_and_class_count():
    from launch import recommend
    small = recommend("r100", "cosface", num_gpus=2, gpu_mem_gb=24, num_classes=90_000, num_image=5_000_000)
    assert small["batch_size"] % 32 == 0 and 32 <= small["batch_size"] < 512
    assert small["sample_rate"] == 1.0 and small["margin_list"] == (1.0, 0.0, 0.4)
    big = recommend("r100", "cosface", num_gpus=4, gpu_mem_gb=141, num_classes=90_000, num_image=5_000_000)
    assert big["batch_size"] >= small["batch_size"]


def test_render_config_is_importable_and_complete():
    from launch import recommend, render_config
    r = recommend("r100", "arcface", 4, 141, 2_059_906, 42_474_557)
    src = render_config(r, rec="/data/webface42m", output="work_dirs/my_run", pretrained="/w/model.pt",
                        resume=False, num_workers=8, dali=True, seed=2048)
    ns = {}
    exec(src, ns)
    cfg = ns["config"]
    for k in ("network", "rec", "num_classes", "num_image", "batch_size", "sample_rate", "lr", "optimizer",
              "num_epoch", "warmup_epoch", "fp16", "dali", "pretrained", "output", "save_all_states",
              "margin_list", "embedding_size", "weight_decay", "num_workers"):
        assert k in cfg, k
    assert cfg.pretrained == "/w/model.pt" and cfg.dali is True and cfg.save_all_states is True
