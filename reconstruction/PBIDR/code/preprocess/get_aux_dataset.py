import os
import sys
sys.path.append(os.path.abspath(''))
import torch
import argparse
import numpy as np
from pytorch3d.io import load_objs_as_meshes, save_obj,load_obj
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    # FoVPerspectiveCameras,
    PointLights,
    # DirectionalLights,
    # Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    # SoftPhongShader,
    # SoftSilhouetteShader,
    SoftPhongShader,
    # TexturesVertex,
    Materials
)
from PIL import Image

print("Start to get aux dataset!")

if __name__ == '__main__':

    parser = argparse.ArgumentParser("PreProcessing")
    parser.add_argument('--gpu', '-g', type=str, default='0',help='GPU')
    parser.add_argument('--input', '-i', type=str, default='../raw_data', help='Location of Raw Textured Mesh Dataset')
    parser.add_argument('--output', '-o', type=int, required=True, help='New aux dataset')
    parser.add_argument('--yaw', type=int, default=15, help='num_views_yaw')
    parser.add_argument('--yaw_angle', type=int, default=45, help='yaw_angle')
    parser.add_argument('--pitch', type=int, default=9, help='num_views_pitch')
    parser.add_argument('--pitch_angle', type=int, default=30, help='pitch_angle')
    parser.add_argument('--datapath', type=str, default='../data/', help='Location of code data')

    parser.add_argument('--dataset', '-d', type=str, default='Face', help='FaceTest dataset')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Setup
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # Set paths
    DATA_DIR = args.input
    IMAGE_DIR = os.path.join(DATA_DIR, "mesh")

    if os.path.exists(IMAGE_DIR):
        os.system("rm -r " + IMAGE_DIR)
    os.mkdir(IMAGE_DIR)
    obj_filename = os.path.join(DATA_DIR, "mesh.obj")

    if not os.path.exists(os.path.join(args.datapath, args.dataset)):
        os.mkdir(os.path.join(args.datapath, args.dataset))

    # Load obj file
    mesh = load_objs_as_meshes([obj_filename], device=device,load_textures=True)
    print(obj_filename)
    print("Loaded Mesh")

    # the number of different viewpoints from which we want to render the mesh.
    def Ry(q):
        return np.array([[-np.cos(q * np.pi / 180), 0, -np.sin(q * np.pi / 180)], [0, 1, 0],
                         [np.sin(q * np.pi / 180), 0, -np.cos(q * np.pi / 180)]])
    def Rx(q):
        return np.array([[-1, 0, 0], [0, np.cos(q * np.pi / 180), np.sin(q * np.pi / 180)],
                         [0, np.sin(q * np.pi / 180), -np.cos(q * np.pi / 180)]])

    def get_R_matrix(azim, axis="Ry"):
        print("Rotation Martix {}".format(axis))
        aa = []
        if axis == "Ry":
            for q in azim:
                aa.append(Ry(q))
            RRR = torch.tensor(np.array(aa)).to(device)
        else:
            for q in azim:
                aa.append(Rx(q))
            RRR = torch.tensor(np.array(aa)).to(device)
        return RRR

    num_views = args.yaw + args.pitch

    yaw_dim = torch.linspace(-1 * args.yaw_angle, args.yaw_angle, args.yaw)
    pitch_dim = torch.linspace(-1 * args.pitch_angle, args.pitch_angle , args.pitch)

    lights = PointLights(device=device, location=[[0, 50, 100]], ambient_color=((1.0, 1.0, 1.0), ), diffuse_color=((0.0, 0.0, 0.0), ), specular_color=((0.0, 0.0, 0.0), ))
    RRy, TTy = look_at_view_transform(dist=8, elev=0, azim=yaw_dim, up=((0, 1, 0),), device=device)

    TTx = TTy[:args.pitch]
    RRx = get_R_matrix(azim=pitch_dim, axis="Rx")

    Rtotal = torch.cat([RRy, RRx], dim=0)
    Ttotal = torch.cat([TTy, TTx], dim=0)

    cameras = PerspectiveCameras(device=device, focal_length=4500, principal_point=((512, 512),), R=Rtotal, T=Ttotal,
                                 image_size=((1024, 1024),))

    if num_views != 1:
        camera = PerspectiveCameras(device=device, focal_length=4500, principal_point=((512, 512),), R=Rtotal[None, 1, ...],
                                T=Ttotal[None, 1, ...], image_size=((1024, 1024),))
    else:
        camera = PerspectiveCameras(device=device, focal_length=4500, principal_point=((512, 512),),
                                    R=Rtotal,
                                    T=Ttotal, image_size=((1024, 1024),))

    mymaterials = Materials(device=device, shininess=8)
    raster_settings = RasterizationSettings(
        image_size=1024,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=camera,
            lights=lights,
            materials=mymaterials,
        )
    )

    meshes = mesh.extend(num_views)
    target_images = renderer(meshes, cameras=cameras, lights=lights)
    target_rgb = [target_images[i, ..., :3] for i in range(num_views)]
    target_cameras = [PerspectiveCameras(device=device, focal_length=4500, principal_point=((512, 512),), R=Rtotal[None, i, ...],
                           T=Ttotal[None, i, ...], image_size=((1024, 1024),)) for i in range(num_views)]

    # RGB images
    if not os.path.exists(os.path.join(IMAGE_DIR, 'image')):
        os.mkdir(os.path.join(IMAGE_DIR, 'image'))
    if not os.path.exists(os.path.join(IMAGE_DIR, 'mask')):
        os.mkdir(os.path.join(IMAGE_DIR, 'mask'))

    for i in range(len(target_images)):
        img = Image.fromarray((target_images[i, ..., :3].cpu().numpy() * 255).astype(np.uint8))
        img.save(os.path.join(IMAGE_DIR, 'image/{0}.png'.format('%03d' % int(i+1))))
        img.save(os.path.join(IMAGE_DIR, 'mask/{0}.png'.format('%03d' % int(i+1))))
    np.save(os.path.join(IMAGE_DIR,'R.npy'), Rtotal.cpu().numpy())
    np.save(os.path.join(IMAGE_DIR,'T.npy'), Ttotal.cpu().numpy())

    SCAN_DIR = args.datapath + args.dataset + '/scan' + str(args.output) + "/"
    if os.path.exists(SCAN_DIR):
        os.system("rm -r " + SCAN_DIR)
    os.system("cp -r " + IMAGE_DIR + " " + SCAN_DIR)
    os.system("cp " + DATA_DIR + "/mesh.* " + SCAN_DIR + ".")
    print("Finished")