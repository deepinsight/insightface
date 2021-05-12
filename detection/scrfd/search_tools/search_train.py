import os, sys



def train(group, prefix, idx, gpuid):
    assert idx>=0
    cmd = "CUDA_VISIBLE_DEVICES='%d' PORT=%d bash ./tools/dist_train.sh ./configs/%s/%s_%d.py 1 --no-validate"%(gpuid,29100+idx, group, prefix, idx)
    print(cmd)
    os.system(cmd)


gpuid = int(sys.argv[1])
idx_from = int(sys.argv[2])
idx_to = int(sys.argv[3])
group = 'scrfdgen'
if len(sys.argv)>4:
    group = sys.argv[4]

for idx in range(idx_from, idx_to):
    train(group, group, idx, gpuid)

