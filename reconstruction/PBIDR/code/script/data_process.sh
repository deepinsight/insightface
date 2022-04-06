set -ex

GPU=0

python preprocess/get_aux_dataset.py -g $GPU -i '../raw_data/0' -o 0 -d 'Test' --yaw 17 --pitch 0
python preprocess/preprocess_cameras.py -i 0 -d 'Test'