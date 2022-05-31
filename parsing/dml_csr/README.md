# Decoupled Multi-task Learning with Cyclical Self-Regulation for Face Parsing.

The official repository of *[Decoupled Multi-task Learning with Cyclical Self-Regulation for Face Parsing. (CVPR 2022)](https://arxiv.org/abs/2203.14448)*. 

## Installation

Our model is based on Pytorch 1.7.1 with Python 3.6.2.

```sh
pip install -r requirements.txt
```

## Data
You can download original datasets:
- **Helen** : [https://www.sifeiliu.net/face-parsing](https://www.sifeiliu.net/face-parsing)
- **LaPa** : [https://github.com/JDAI-CV/lapa-dataset](https://github.com/JDAI-CV/lapa-dataset)
- **CelebAMask-HQ** : [https://github.com/switchablenorms/CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)

and put them in ./dataset folder as below
```
dataset/
    images/
    labels/
    edges/
    train_list.txt
    test_list.txt
        each line: 'images/100032540_1.jpg labels/100032540_1.png'
```
Besides, we provide the edge genearation code in the *generate_edge.py*.

## Usage

If you need imagenet pretrained resent-101, please download from [baidu drive]() or [Google drive](https://drive.google.com/open?id=1rzLU-wK6rEorCNJfwrmIu5hY2wRMyKTK), and put it into snapshot folder.

For dstributed(multi-gpu) training. Inplace-abn requires pytorch distributed data parallel.
```
GPU=4,5,6,7
Node=4
dataset=./datasets/CelebAMask-HQ/
snapshot=./work_dirs/
CUDA_VISIBLE_DEVICES="$GPU" python -m torch.distributed.launch --nproc_per_node="$Node"  --master_port=295002 train.py --data-dir "$dataset"  --random-mirror --random-scale \
--gpu "$GPU" --batch-size 7 --input-size 473,473 --snapshot-dir "$snapshot" --num-classes 19 --epochs 200 --schp-start 150
```

For testing [pretrained models](https://drive.google.com/file/d/1-PjUts1AMzXNyvw3VaJQmg43GJbfEpEQ/view?usp=sharing)
```
python test.py --data-dir "$dataset" --out-dir "$out_dir" --restore-from "$snapshot" --gpu "$GPU" --batch-size 7 --input-size 473,473 --dataset test --num-classes 19
```

## Reference

If you consider use our code, please cite our paper:

```
@inproceedings{Zheng2022DecoupledML,
  title={Decoupled Multi-task Learning with Cyclical Self-Regulation for Face Parsing},
  author={Qi Zheng and Jiankang Deng and Zheng Zhu and Ying Li and Stefanos Zafeiriou},
  booktitle={Computer Vision and Pattern Recognition},
  year={2022}
}
```
