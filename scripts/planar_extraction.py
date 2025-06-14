from load_assets import *
import sys 
import argparse
from collections import defaultdict

s3dis_labels = get_s3dis_labels()

'''
save_planar_clouds extracts 4 labels of clouds (walls, floors, ceiling, beam, column, window, door, and others) 
from the given point cloud and stores them individually in the output directory.
'''
def save_planar_clouds(out_path, pc, pipeline_r):
    print("Running Inference...")
    results_r = pipeline_r.run_inference(pc)
    print("Done! Segmented Following S3DIS Labels: " + str(set(results_r['predict_labels'])))

    pts = pc['point']
    feature = pc['feat']
    
    # create pts and save ply files
    pts_groups, feat_groups = defaultdict(list), defaultdict(list); [[pts_groups[min(label, 7)].append(pts[j]), feat_groups[min(label, 7)].append(feature[j])] for j, label in enumerate(results_r['predict_labels'])]
    filenames = ['ceiling.ply', 'floor.ply', 'walls.ply', 'beam.ply', 'column.ply', 'window.ply', 'door.ply', 'others.ply']
    [o3d.io.write_point_cloud(os.path.join(out_path, filenames[i]), open3d_pcd(pts_groups[i], feat_groups[i])[0]) for i in range(8) if len(pts_groups[i]) != 0]; print("Saved the PLY files!")
    print("Saved the PLY files!")

def main(data_path, out_path):
    os.makedirs(out_path, exist_ok = True)
    pipeline = get_pipeline()
    pcs = get_custom_data(data_path)
    save_planar_clouds(out_path, pcs, pipeline)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extraction of planar elements from the results of 3D indoor Segmentation with RandLANet model and saving into their respective ply files")
    parser.add_argument("--data_path", help="The path to input PLY mesh file")
    parser.add_argument("--output_dir", help="The path to store floor plan images")
    args = parser.parse_args()

    input_path = args.data_path
    output_path = args.output_dir

    if not args.data_path or not args.output_dir:
        parser.print_help(sys.stderr)
        sys.exit(1)

    main(input_path, output_path)