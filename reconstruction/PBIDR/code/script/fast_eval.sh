set -ex

GPU=0
python evaluation/eval.py --conf ./confs/test.conf --scan_id 0 --gpu $GPU --checkpoint 400 --eval_rendering