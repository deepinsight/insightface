set -ex

GPU=0
python training/runner.py --conf ./confs/test.conf --scan_id 0 --gpu $GPU --nepoch 400