import os
from os.path import exists, join
from plyfile import PlyData, PlyElement
import numpy as np
import open3d as o3d

def load_model():
  ckpt_folder = "./logs"
  randlanet_url = "https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_s3dis_202201071330utc.pth"
  ckpt_path = ckpt_folder + "/vis_weights_{}.pth".format('RandLANet')
  if not exists(ckpt_path):
    os.makedirs(ckpt_folder, exist_ok = True) 
    cmd = "wget {} -O {}".format(randlanet_url, ckpt_path)
    os.system(cmd)
    print("Pretrained RandLANet weight download success")
  print("INFO: Found checkpoint----RandLANet")
  return ckpt_path


def get_custom_data(pc_path):

    data = PlyData.read(pc_path)['vertex']

    point = np.zeros((data['x'].shape[0], 3), dtype=np.float32)
    point[:, 0] = data['x']
    point[:, 1] = data['y']
    point[:, 2] = data['z']

    feat = np.zeros(point.shape, dtype=np.float32)
    fields = data.data.dtype.names

    if 'red' in fields and 'green' in fields and 'blue' in fields:
        feat[:, 0] = data.data['red']
        feat[:, 1] = data.data['green']
        feat[:, 2] = data.data['blue']
    elif 'diffuse_red' in fields and 'diffuse_green' in fields and 'diffuse_blue' in fields:
        feat[:, 0] = data.data['diffuse_red']
        feat[:, 1] = data.data['diffuse_green']
        feat[:, 2] = data.data['diffuse_blue']
    else:
        print("No Color Information Detected! Recheck the .PLY file for 'red', 'green', 'blue' or 'diffuse_red', 'diffuse_green', 'diffuse_blue'.")


    data = {
        'point': point,
        'feat': feat,
        'label': np.zeros(len(point)),          
    }

    return data

def open3d_pcd(pts, feat):
    pts = np.asarray(pts, dtype=np.float64)
    feat = np.asarray(feat, dtype=np.uint8)
    feat = feat/ 255.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(feat)
    return pcd, pts, feat