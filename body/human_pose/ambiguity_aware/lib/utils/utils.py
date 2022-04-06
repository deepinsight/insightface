import os
import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import auc
        
joint_parents = [1, 2, 13, 13, 3, 4, 7, 8, 12, 12, 9, 10, 14, 13, 13, 12, 15]

def rigid_align(predicted, target):
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t
    # Return MPJPE
    return predicted_aligned


def p_mpjpe(predicted, target, rot=True, trans=True, scale=True):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    if rot: 
        predicted_aligned = np.matmul(predicted, R)
    else: 
        predicted_aligned = predicted

    if scale:
        predicted_aligned = a * predicted_aligned 
    if trans: 
        predicted_aligned = predicted_aligned + t
    # predicted_aligned = a * np.matmul(predicted, R) + t

    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1))


def get_rotation_y_v2(angle, is_mpi=False):
    # first get the rod matrix 
    bs = angle.size(0)
    cos, sin = torch.cos(angle), torch.sin(angle)
    cos = cos.repeat(3, 1).view(3, bs).permute(1, 0).contiguous().view(-1, 1)
    sin = sin.repeat(3, 1).view(3, bs).permute(1, 0).contiguous().view(-1, 1)
    # if is_mpi: 
    #     rx, ry, rz = -0.3189, 0.3282, 0.8891
    # else:
    rx, ry, rz = -0.01474, 0.96402, 0.261718 
    # rx, ry, rz = 0, 1, 0
    r = torch.tensor([[rx, ry, rz]])
    r_mat = r.t().matmul(r)
    r_hat = torch.tensor([[0, -rz, ry], [rz, 0, -rx], [-ry, rx, 0]])
    e1 = cos * torch.eye(3).repeat(bs, 1).type_as(cos) 
    e2 = (1 - cos) * r_mat.repeat(bs, 1).type_as(cos)
    e3 = sin * r_hat.repeat(bs, 1).type_as(sin)
    mat = e1 + e2 + e3
    mat = mat.view(bs, 3, 3)
    return mat 
    
def get_rotation_y(angle):
    bs = angle.size(0)
    sin, cos = torch.sin(angle), torch.cos(angle)
    mat = torch.zeros((bs * 3, 3)).type_as(sin)
    mat[0:bs, 0:1], mat[0:bs, 2:3] = cos, sin
    mat[bs:2*bs, 1] = 1.0
    mat[bs*2:bs*3, 0:1], mat[bs*2:bs*3, 2:3] = -sin, cos 
    mat = mat.view(3, bs, 3).permute(1, 0, 2)
    return mat 

def get_rotation_x(angle):
    bs = angle.size(0)
    sin, cos = torch.sin(angle), torch.cos(angle)
    mat = torch.zeros((bs * 3, 3)).type_as(sin)
    mat[0:bs, 0] = 1.0
    mat[bs:bs*2, 1:2], mat[bs:bs*2, 2:3] = cos, -sin
    mat[bs*2:bs*3, 1:2], mat[bs*2:bs*3, 2:3] = sin, cos 
    mat = mat.view(3, bs, 3).permute(1, 0, 2)
    return mat 

def get_rotation_z(angle):
    bs = angle.size(0)
    sin, cos = torch.sin(angle), torch.cos(angle)
    mat = torch.zeros((bs * 3, 3)).type_as(sin)
    mat[2*bs:3*bs, 2] = 1.0 
    mat[0:bs, 0:1], mat[0:bs, 1:2] = cos, -sin 
    mat[bs:2*bs, 0:1], mat[bs:2*bs, 1:2] = sin, cos
    mat = mat.view(3, bs, 3).permute(1, 0, 2)
    return mat 

def euler2rotmat(eulers):
    # inputs' shape: (N, 3), tensors
    # rotate in the order of z, x, y
    n = eulers.size(0)
    thetax, thetay, thetaz = eulers[:, 0:1], eulers[:, 1:2], eulers[:, 2:3]
    matx = get_rotation_x(thetax)
    maty = get_rotation_y(thetay)
    matz = get_rotation_z(thetaz)
    rotmat = matz.matmul(matx).matmul(maty)
    # rotmat = maty.matmul(matx).matmul(matz)
    return rotmat

def rotate(joints_3d, eulers):
    rotmat = euler2rotmat(eulers)
    root = joints_3d[:, 13:14] if joints_3d.shape[1] == 17 else joints_3d[:, 12:13]
    joints_3d = joints_3d - root
    joints_3d = joints_3d.matmul(rotmat)
    # joints_3d = rotmat.matmul(joints_3d.permute(0, 2, 1))
    # joints_3d = joints_3d.permute(0, 2, 1).contiguous()
    joints_3d = joints_3d + root
    return joints_3d

def rotate2(joints_3d, rotmat):
    n = rotmat.size(0)
    rotmat = rotmat.view(n, 3, 3)
    joints_3d = joints_3d.matmul(rotmat)
    # joints_3d = rotmat.matmul(joints_3d.permute(0, 2, 1)).permute(0, 2, 1)
    return joints_3d

def transform_3d(inputs, rot_y, rot_x, is_reverse):
    # rot_y/rot_x: N x 1 
    root3d = inputs[:,13:14].clone() if inputs.shape[1] == 17 else inputs[:, 12:13]
    outputs = inputs - root3d 
    rot_y_mat = get_rotation_y_v2(rot_y)
    rot_x_mat = get_rotation_x(rot_x)
    rot_mat = rot_x_mat.bmm(rot_y_mat) if not is_reverse else rot_y_mat.bmm(rot_x_mat)
    # N x 3 x 3 , ((N x J x 3) -> (N x 3 x J)) -> N x 3 x J
    outputs = rot_mat.bmm(outputs.permute(0, 2, 1))
    outputs = outputs.permute(0, 2, 1)
    outputs += root3d
    return outputs

# this is the transformation defined by the paper originally 
def transform_3d_v2(inputs, rot_y, rot_x, shift, is_reverse, rot_z=None, use_new_rot=False, is_mpi=False):
    shift = torch.FloatTensor([0.0, 0.0, shift]).type_as(inputs).view(1, 1, 3)
    shift = shift.expand_as(inputs)
    if is_reverse:
        outputs = inputs - shift 
    else:
        root3d = inputs[:, 13:14].clone() if inputs.shape[1] == 17 else inputs[:, 12:13]
        outputs = inputs - root3d 
    if use_new_rot: 
        rot_y_mat = get_rotation_y_v2(rot_y, is_mpi=is_mpi)
    else:
        rot_y_mat = get_rotation_y(rot_y)
    rot_x_mat = get_rotation_x(rot_x)
    if rot_z is None:
        rot_mat = rot_x_mat.bmm(rot_y_mat) if not is_reverse else rot_y_mat.bmm(rot_x_mat)
    else: 
        rot_z_mat = get_rotation_z(rot_z)
        rot_mat = rot_z_mat.bmm(rot_x_mat).bmm(rot_y_mat) if not is_reverse else rot_y_mat.bmm(rot_x_mat).bmm(rot_z_mat)
    # N x 3 x 3 , ((N x J x 3) -> (N x 3 x J)) -> N x 3 x J
    outputs = rot_mat.bmm(outputs.permute(0, 2, 1))
    outputs = outputs.permute(0, 2, 1)
    # add the shift instead of the root 
    if not is_reverse: 
        outputs += shift 
    return outputs


def Transform3DV1(inputs, rot_x, rot_y, isReverse):
    ## shape of inputs ==> (B, J, 3)
    ## shape of rot_x, rot_y ==> (B, 1)
    root3d = inputs[:,13:14].clone() if inputs.shape[1] == 17 else inputs[:, 12:13]
    inputs -= root3d 
    if not isReverse:
        ## first 3D poses are rotated with Y axis
        inputs_ = inputs.clone()
        inputs_[:,:,0] = torch.cos(rot_x) * inputs[:,:,0] + torch.sin(rot_x) * inputs[:,:,2]
        inputs_[:,:,2] = -torch.sin(rot_x) * inputs[:,:,0] + torch.cos(rot_x) * inputs[:,:,2]
        ## second 3D poses are rotated with X axis
        inputs = inputs_.clone()
        inputs[:,:,1] = torch.cos(rot_y) * inputs_[:,:,1] - torch.sin(rot_y) * inputs_[:,:,2]
        inputs[:,:,2] = torch.sin(rot_y) * inputs_[:,:,1] + torch.cos(rot_y) * inputs_[:,:,2]
    else:
        ## first 3D poses are rotated with X axis
        inputs_ = inputs.clone()
        inputs_[:,:,1] = torch.cos(rot_y) * inputs[:,:,1] - torch.sin(rot_y) * inputs[:,:,2]
        inputs_[:,:,2] = torch.sin(rot_y) * inputs[:,:,1] + torch.cos(rot_y) * inputs[:,:,2]
        ## second 3D poses are rotated with Y axis
        inputs = inputs_.clone()
        inputs[:,:,0] = torch.cos(rot_x) * inputs_[:,:,0] + torch.sin(rot_x) * inputs_[:,:,2]
        inputs[:,:,2] = -torch.sin(rot_x) * inputs_[:,:,0] + torch.cos(rot_x) * inputs_[:,:,2]
    inputs += root3d
    return inputs
  
def Transform3D(inputs, rot_x, rot_y, isReverse):
    ## shape of inputs ==> (B, J, 3)
    ## shape of rot_x, rot_y ==> (B, 1)
    root3d = inputs[:,13:14].clone() if inputs.shape[1] == 17 else inputs[:, 12:13]
    inputs -= root3d
    rot_x = rot_x.unsqueeze(-1)
    rot_y = rot_y.unsqueeze(-1)
    if not isReverse:
        ## first 3D poses are rotated with Y axis
        inputs = \
        torch.cat([torch.cos(rot_x) * inputs[:,:,0:1] - torch.sin(rot_x) * inputs[:,:,2:3],
                   inputs[:,:,1:2],
                   torch.sin(rot_x) * inputs[:,:,0:1] + torch.cos(rot_x) * inputs[:,:,2:3]], -1)
        
        ## second 3D poses are rotated with X axis
        inputs = \
        torch.cat([inputs[:,:,0:1], 
                   torch.cos(rot_y) * inputs[:,:,1:2] + torch.sin(rot_y) * inputs[:,:,2:3],
                   -torch.sin(rot_y) * inputs[:,:,1:2] + torch.cos(rot_y) * inputs[:,:,2:3]], -1)
    else:
        ## first 3D poses are rotated with X axis
        inputs = \
        torch.cat([inputs[:,:,0:1], 
                   torch.cos(rot_y) * inputs[:,:,1:2] + torch.sin(rot_y) * inputs[:,:,2:3],
                   -torch.sin(rot_y) * inputs[:,:,1:2] + torch.cos(rot_y) * inputs[:,:,2:3]], -1)
        ## second 3D poses are rotated with Y axis
        inputs = \
        torch.cat([torch.cos(rot_x) * inputs[:,:,0:1] - torch.sin(rot_x) * inputs[:,:,2:3],
                   inputs[:,:,1:2],
                   torch.sin(rot_x) * inputs[:,:,0:1] + torch.cos(rot_x) * inputs[:,:,2:3]], -1)
    inputs += root3d
    return inputs

def get_optimizer(cfg, model, is_dis=False, is_temp=False):
    optimizer = None
    if is_dis: 
        lr = cfg.TRAIN.TEMP_LR if is_temp else cfg.TRAIN.DIS_LR 
    else: 
        lr = cfg.TRAIN.POSE_LR
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr, 
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(0.5, 0.9)
        )

    return optimizer

def load_checkpoint(model, optimizer, output_dir, filename='checkpoint.pth.tar'):
    file = os.path.join(output_dir, filename)
    if os.path.isfile(file):
        checkpoint = torch.load(file)
        start_epoch = checkpoint['epoch']
        model.module.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('=> load checkpoint {} (epoch {})'
              .format(file, start_epoch))

        return start_epoch, model, optimizer

    else:
        print('=> no checkpoint found at {}'.format(file))
        return 0, model, optimizer


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth.tar'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best: 
        torch.save(states, os.path.join(output_dir, "model_best.pth.tar"))

def test_transform_3d():
    x1 = torch.randn(100, 17, 3)
    x2 = x1.clone()
    rot_y = torch.randn(100, 1)
    rot_x = torch.randn(100, 1)
    is_reverse = True
    r1 = transform_3d(x1, rot_y, rot_x, is_reverse)
    r2 = Transform3DV1(x2, rot_y, rot_x, is_reverse)
    if (r1 - r2 < 1e-5).all():
        print("Passed.")
    else:
        raise ValueError("At least one computation is not corrected")

def test_transform_3d_v2():
    bs = 100
    x1 = torch.randn(bs, 17, 3)
    rot_y = torch.randn(bs, 1)
    rot_x = torch.randn(bs, 1)
    rot_z = torch.randn(bs, 1)
    root = x1[:, 13:14] if x1.shape[1] == 17 else x1[:, 12:13]
    x2 = transform_3d_v2(x1, rot_y, rot_x, 10.0, False, rot_z, use_new_rot=True)
    x22 = transform_3d_v2(x1, rot_y, rot_x, 10.0, False, rot_z, use_new_rot=False)
    print((x22 - x2 < 1e-5).all().item())
    x3 = transform_3d_v2(x2, -rot_y, -rot_x, 10.0, True, -rot_z, use_new_rot=True)
    x3 = x3 + root 
    print((x1 - x3 < 1e-5).all().item())

def test_p_mpjpe():
    x = np.random.randn(32, 17, 3)
    y = np.random.randn(32, 17, 3)
    err = p_mpjpe(x, y)
    print(err)

def get_pck3d(joints_3d_pre, joints_3d_gt):
    # about half of the head size 
    threshold = 150 / 2048 
    n, c, _ = joints_3d_pre.shape
    cnt = (np.linalg.norm(joints_3d_pre - joints_3d_gt, axis=-1) < threshold).sum()
    return cnt / (n * c)

def _scale_range(x, a, b): 
    m = x.min()
    M = x.max()
    return (x - M)/(m - M)*(a - b) + b

def calc_dists(joints_3d_pre, joints_3d_gt, head_size=300):
    dists = 1000 / head_size * np.linalg.norm(joints_3d_pre - joints_3d_gt, axis=-1)
    return dists

def calc_pck3d(dists, threshold=0.5):
    n, c = dists.shape
    return (dists < threshold).sum() / (n * c)

def calc_auc(joints_3d_pre, joints_3d_gt, head_size=300, is_mpi=True):
    indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16]
    # joints_3d_pre = joints_3d_pre[:, indices]
    # joints_3d_gt = joints_3d_gt[:, indices]
    dists = calc_dists(joints_3d_pre, joints_3d_gt, head_size=head_size)
    x = np.arange(0.0, 0.51, 0.05)
    pcks = []
    pckh5 = 0.0
    for thresh in x: 
        pck = calc_pck3d(dists, thresh)
        if thresh == 0.50: 
            pckh5 = pck
        pcks.append(pck)
    # scale to 0~1
    x = _scale_range(x, 0, 1)
    auc_val = auc(x, np.array(pcks))
    # the second output is the pckh@0.5
    return auc_val, pckh5

def calc_auc_aligned(joints_3d_pre, joints_3d_gt, head_size=300, is_mpi=True):
    joints_3d_pre = rigid_align(joints_3d_pre, joints_3d_gt)
    indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16]
    # joints_3d_pre = joints_3d_pre[:, indices]
    # joints_3d_gt = joints_3d_gt[:, indices]
    dists = calc_dists(joints_3d_pre, joints_3d_gt, head_size=head_size)
    x = np.arange(0.0, 0.51, 0.05)
    pcks = []
    pckh5 = 0.0
    for thresh in x: 
        pck = calc_pck3d(dists, thresh)
        if thresh == 0.50: 
            pckh5 = pck
        pcks.append(pck)
    # scale to 0~1
    x = _scale_range(x, 0, 1)
    auc_val = auc(x, np.array(pcks))
    # the second output is the pckh@0.5
    return auc_val, pckh5

if __name__ == "__main__":
    # test_transform_3d_v2()
    # test_p_mpjpe()
    x = torch.tensor([1.5708, 1.5708, -1.5708]).view(1, -1)
    print(euler2rotmat(x))
