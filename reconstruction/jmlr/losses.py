import torch
from torch import nn
import torch.nn.functional as F
import kornia
import numpy as np

#def loss_l1(a, b):
    #_loss = torch.abs(a - b)
    #_loss = torch.mean(_loss, dim=1)
    ##if epoch>4 and cfg.loss_hard:
    ##    _loss, _ = torch.topk(_loss, k=int(cfg.batch_size*0.3))
    #_loss = torch.mean(_loss)
    #return _loss



def loss_pip(outputs_map, outputs_local_x, outputs_local_y, labels_map, labels_local_x, labels_local_y):

    tmp_batch, tmp_channel, tmp_height, tmp_width = outputs_map.size()
    labels_map = labels_map.view(tmp_batch*tmp_channel, -1)
    labels_max_ids = torch.argmax(labels_map, 1)
    labels_max_ids = labels_max_ids.view(-1, 1)

    #print('TTT:', outputs_local_x.shape, tmp_batch, tmp_channel)

    outputs_local_x = outputs_local_x.reshape(tmp_batch*tmp_channel, -1)
    outputs_local_x_select = torch.gather(outputs_local_x, 1, labels_max_ids)
    outputs_local_y = outputs_local_y.reshape(tmp_batch*tmp_channel, -1)
    outputs_local_y_select = torch.gather(outputs_local_y, 1, labels_max_ids)

    labels_local_x = labels_local_x.view(tmp_batch*tmp_channel, -1)
    labels_local_x_select = torch.gather(labels_local_x, 1, labels_max_ids)
    labels_local_y = labels_local_y.view(tmp_batch*tmp_channel, -1)
    labels_local_y_select = torch.gather(labels_local_y, 1, labels_max_ids)

    labels_map = labels_map.view(tmp_batch, tmp_channel, tmp_height, tmp_width)
    loss_map = F.mse_loss(outputs_map, labels_map)
    loss_x = F.l1_loss(outputs_local_x_select, labels_local_x_select)
    loss_y = F.l1_loss(outputs_local_y_select, labels_local_y_select)
    return loss_map, loss_x, loss_y

def eye_like(x: torch.Tensor, n: int) -> torch.Tensor:
    return torch.eye(n, n, dtype=x.dtype, device=x.device).unsqueeze(0).repeat(x.shape[0], 1, 1)

class ProjectLoss(nn.Module):
    
    def __init__(self,M_proj):
        super(ProjectLoss, self).__init__()
        img_w = 800
        img_h = 800
        M1 = np.array([
            [img_w/2,       0, 0, 0],
            [      0, img_h/2, 0, 0],
            [      0,       0, 1, 0],
            [img_w/2, img_h/2, 0, 1]
        ])
        M = M_proj @ M1
        M = M.astype(np.float32)
        self.register_buffer('M', torch.from_numpy(M))

        camera_matrix =  M[:3,:3].T.copy()
        camera_matrix[0,2] = 400
        camera_matrix[1,2] = 400
        camera_matrix[2,2] = 1
        intrinsics = np.array([camera_matrix]).astype(np.float64)
        self.register_buffer('intrinsics', torch.from_numpy(intrinsics))


        self.eps = 1e-5
        #self.projector = Reprojector(img_w,img_h,M_proj)
        #self.solver = PnPSolver(self.projector.M.numpy())
        #self.loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)
        #self.loss_fn = torch.nn.MSELoss()
        self.loss_fn = torch.nn.L1Loss()
        
    
    def forward(self,verts3d, points2d, affine):
        # pred_2d_lmks  Batch*N*2
        # verts Batch*N*3
        ones = torch.ones([points2d.shape[0] , points2d.shape[1], 1],device=points2d.device)
        verts_homo = torch.cat((points2d, ones), 2)
        K  = eye_like(affine,3)
        K[:,:2,:3] = affine
        inv_k = torch.linalg.inv(K)
        inv_k@verts_homo.permute(0,2,1)
        points2d_inv = inv_k@verts_homo.permute(0,2,1)
        points2d_inv = points2d_inv.permute(0,2,1)[:,:,:2]

        intrinsics = self.intrinsics.repeat([verts3d.shape[0],1,1 ])
        #print(verts3d.double().shape) 
        #print(points2d.double().shape)
        #print(intrinsics.shape)
        RT_ = kornia.geometry.solve_pnp_dlt(verts3d.double(), points2d_inv.double(), intrinsics,svd_eps=self.eps)
        RT_ = RT_.float()
        RT = eye_like(verts3d,4)
#         RT[:,1:3,:] *=-1
        RT[:,:3,:] = RT_
        RT = RT.permute(0,2,1)
        RT[:,:,:2] *= -1

        ones = torch.ones([verts3d.shape[0] , verts3d.shape[1], 1],device=verts3d.device)
        verts_homo = torch.cat((verts3d, ones), 2)
        M = self.M.repeat([verts3d.shape[0],1,1 ])
        verts = verts_homo @ RT @ M
        w_ = verts[:,:, [3]]
        verts = verts / w_
        reproject_points2d = verts[:,:, :2]
        loss = self.loss_fn(reproject_points2d , points2d_inv)

        return loss

