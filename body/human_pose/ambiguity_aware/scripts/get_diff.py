import numpy as np 
import h5py 
import argparse 

np.random.seed(2019)
parser = argparse.ArgumentParser(description="Generate the diff data")
parser.add_argument("--valid", action="store_true")
parser.add_argument("--use_random", action="store_true")
# specify the interval 
parser.add_argument("--bound", default=1, type=int, required=False)
parser.add_argument("--use_previous", action="store_true", help="Specify whether to use previous frames or not")
parser.add_argument('--use_pre', action='store_true')
args = parser.parse_args()

# compute the difference of frames 
bound = args.bound
is_train = not args.valid
use_random = args.use_random
use_previous = args.use_previous
use_pre = args.use_pre
# in_filename = "../data/kinetics_final.h5"
suffix = str(bound) if bound > 1 else ""
if use_random: 
    suffix += "_rand"
if use_pre: 
    suffix += "_pre"

in_filename = "../data/h36m_{}_pred3.h5".format("train" if is_train else "valid")
out_filename = "../data/h36m_{}_diff{}.h5".format("train" if is_train else "valid", suffix)

f = h5py.File(in_filename, "r")
names = [name.decode() for name in f['imagename'][:]]
joints_2d = np.array(f['joint_2d_gt' if not use_pre else "joint_2d_pre"])
f.close()
print("Load from", in_filename)

size = joints_2d.shape[0]
splits = [name.split('/') for name in names]
sequences = ['/'.join(split[:3]) for split in splits]
indices = [int(split[-1]) for split in splits]

# calculate the length of each sequence
seq_lens = {}
for split in splits: 
    seq = '/'.join(split[:3])
    if seq not in seq_lens: 
        seq_lens[seq] = 0 
    seq_lens[seq] += 1

intervals = np.random.randint(1, bound + 1, (size, ))
if not use_random:
    intervals.fill(bound)

if use_previous: 
    spec_indices = [i for i, index in enumerate(indices) if index < intervals[i]]
    diff_indices = np.arange(0, size, 1) - intervals
    diff_indices[spec_indices] += 2 * intervals[spec_indices]
else: 
    spec_indices = [i for i, index in enumerate(indices) if index >= seq_lens[sequences[i]] - intervals[i]]
    diff_indices = np.arange(0, size, 1) + intervals
    diff_indices[spec_indices] -= 2 * intervals[spec_indices]

# before_joints = np.concatenate((joints_2d[:1].copy(), joints_2d[:-1].copy()), axis=0)
# after_joints = np.concatenate((joints_2d[1:].copy(), joints_2d[-1:].copy()), axis=0)
# print(before_joints.shape, after_joints.shape)

# diff_before = joints_2d - before_joints
# diff_after = joints_2d - after_joints
# diff_before, diff_after = before_joints, after_joints
# diff_before, diff_after = diff_before[:, np.newaxis], diff_after[:, np.newaxis]

# finally process the special cases 
# diff_before[start_indices] = diff_after[start_indices]
# diff_after[end_indices] = diff_before[end_indices]

# diff = np.concatenate((diff_before, diff_after), axis=1)
# print(diff.shape)

# diff_types = np.ones((len(diff), ), dtype=np.uint8)
# diff_types[start_indices] = 0
# diff_types[end_indices] = 2

diff = joints_2d[diff_indices]
dist = np.linalg.norm((joints_2d - diff).reshape(size, -1), axis=1).mean()
print("Mean distance bewteen diff and original: {:.3f}".format(dist))

f = h5py.File(out_filename, "w")
f['gt_diff'] = diff 
# f['gt_diff_type'] = diff_types
f.close()
print("Saved to", out_filename)
