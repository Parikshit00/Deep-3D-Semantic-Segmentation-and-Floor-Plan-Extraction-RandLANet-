import os
from os.path import exists, join
from plyfile import PlyData, PlyElement
import numpy as np
import open3d as o3d
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d

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

def get_s3dis_labels():
   s3dis_labels = {
    0: 'unlabeled',
    1: 'ceiling',
    2: 'floor',
    3: 'wall',
    4: 'beam',
    5: 'column',
    6: 'window',
    7: 'door',
    8: 'table',
    9: 'chair',
    10: 'sofa',
    11: 'bookcase',
    12: 'board',
    13: 'clutter'
    }
   return s3dis_labels

def get_pipeline():
    cfg_file = "../configs/randlanet_s3dis.yml"
    ckpt_path = load_model()
    cfg = _ml3d.utils.Config.load_from_file(cfg_file)
    cfg.model.ckpt_path = ckpt_path
    model = ml3d.models.RandLANet(**cfg.model)
    pipeline = ml3d.pipelines.SemanticSegmentation(model, **cfg.pipeline)
    pipeline.load_ckpt(model.cfg.ckpt_path)
    return pipeline

def ml3d_visualizer(pcs_with_pred):
    s3dis_labels = get_s3dis_labels()
    v = ml3d.vis.Visualizer()
    lut = ml3d.vis.LabelLUT()
    for val in sorted(s3dis_labels.keys()):
        lut.add_label(s3dis_labels[val], val)
    v.set_lut("pred",lut)
    v.visualize(pcs_with_pred)

def open3d_visuaizer(pcs):
    o3d.visualization.draw_geometries(pcs, zoom=0.8, front=[-0.4999, -0.1659, -0.8499], lookat=[2.1813, 2.0619, 2.0999], up=[0.1204, -0.9852, 0.1215])

