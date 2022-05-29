# Face_Detection_Alignment
Face Detection and Alignment Tool
3D projection landmarks (84) and 2D multi-view landmarks(39/68)

Environment:
Tensorflow 1.3, menpo, python 3.5

Train:
CUDA_VISIBLE_DEVICES="1" python train.py --train_dir=ckpt/3D84 --batch_size=8 --initial_learning_rate=0.0001 --dataset_dir=3D84/300W.tfrecords,3D84/afw.tfrecords,3D84/helen_testset.tfrecords,3D84/helen_trainset.tfrecords,3D84/lfpw_testset.tfrecords,3D84/lfpw_trainset.tfrecords,3D84/ibug.tfrecords,3D84/menpo_trainset.tfrecords --n_landmarks=84

Test:
3D model: 84
2D model: frontal68/Union68/Union86(better)

Pretrained Models:
https://drive.google.com/open?id=1DKTeRlJjyo_tD1EluDjYLhtKFPJ9vIVd
