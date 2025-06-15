import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import math


PCD_FILE_PATH = "pcd_floorplan.ply"

SLICE_HEIGHT = 1.5      # Height in meters for the cross-section (e.g., eye-level).
SLICE_THICKNESS = 0.2   # Thickness of the slice (e.g., 20cm) to ensure we capture walls.
GRID_RESOLUTION = 0.02  # Size of each pixel in the 2D grid in meters (2cm).

# -- Line Detection and Refinement
LSD_MIN_LINE_LENGTH = 30  # Minimum line length in pixels for the Line Segment Detector.
MERGE_ANGLE_TOLERANCE = 5 # Degrees. Lines within this angle difference are considered for merging.
MERGE_DISTANCE_TOLERANCE = 15 # Pixels. Perpendicular distance tolerance for merging lines.


def create_placeholder_pcd():
    """Creates a simple box-shaped point cloud if the primary file is not found."""
    print("PCD file not found. Creating a placeholder hollow box for demonstration.")
    points = []
    for x in np.arange(0, 5.0, 0.05):
        for z in np.arange(0, 2.5, 0.05):
            points.append([x, 0, z]) 
            points.append([x, 8, z]) 
    for y in np.arange(0, 8.0, 0.05):
        for z in np.arange(0, 2.5, 0.05):
            points.append([0, y, z]) 
            points.append([5, y, z])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.6, 0.6, 0.6])
    return pcd

def create_occupancy_grid(points_2d, resolution):
    """Converts 2D points into a clean binary occupancy grid image."""
    if points_2d.shape[0] == 0:
        print("Warning: No points found in the slice to create an occupancy grid.")
        return None, None, None

    min_coord = points_2d.min(axis=0)
    max_coord = points_2d.max(axis=0)

    grid_size = np.ceil((max_coord - min_coord) / resolution).astype(int)
    occupancy_grid = np.zeros(grid_size[::-1], dtype=np.uint8) 

    grid_coords = ((points_2d - min_coord) / resolution).astype(int)
    grid_coords[:, 0] = np.clip(grid_coords[:, 0], 0, grid_size[0] - 1)
    grid_coords[:, 1] = np.clip(grid_coords[:, 1], 0, grid_size[1] - 1)
    
    occupancy_grid[grid_coords[:, 1], grid_coords[:, 0]] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grid_processed = cv2.morphologyEx(occupancy_grid, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return grid_processed, min_coord, resolution

def merge_lines(lines, angle_thresh_deg, dist_thresh):
    """Merges collinear and nearby line segments. Expects a NumPy array."""
    if lines.size == 0:
        return []

    lines_rad = [(lines[i], np.arctan2(lines[i,3]-lines[i,1], lines[i,2]-lines[i,0])) for i in range(len(lines))]
    merged_lines = []
    
    while lines_rad:
        base_line, base_angle = lines_rad.pop(0)
        current_points = list(base_line.reshape(2, 2))
        
        i = 0
        while i < len(lines_rad):
            comp_line, comp_angle = lines_rad[i]
            angle_diff = abs(math.degrees(base_angle - comp_angle))
            angle_diff = min(angle_diff, 180 - angle_diff)

            if angle_diff < angle_thresh_deg:
                p1 = np.array([comp_line[0], comp_line[1]])
                p_base_1 = np.array([base_line[0], base_line[1]])
                p_base_2 = np.array([base_line[2], base_line[3]])
                
                line_vec = p_base_2 - p_base_1
                point_vec = p1 - p_base_1
                
                line_len_sq = np.dot(line_vec, line_vec)
                if line_len_sq == 0:
                    i += 1
                    continue
                    
                cross_product = np.cross(point_vec, line_vec)
                d = np.linalg.norm(cross_product) / np.linalg.norm(line_vec)
                
                if d < dist_thresh:
                    current_points.extend(list(comp_line.reshape(2, 2)))
                    lines_rad.pop(i)
                else:
                    i += 1
            else:
                i += 1
        
        points_array = np.array(current_points, dtype=np.int32)
        vx, vy, x, y = cv2.fitLine(points_array, cv2.DIST_L2, 0, 0.01, 0.01)
        
        direction_vec = np.array([vx[0], vy[0]])
        origin_pt = np.array([x[0], y[0]])
        
        projections = [np.dot(pt - origin_pt, direction_vec) for pt in points_array]
        
        min_proj, max_proj = min(projections), max(projections)
        
        p1_new = origin_pt + min_proj * direction_vec
        p2_new = origin_pt + max_proj * direction_vec

        merged_lines.append([p1_new[0], p1_new[1], p2_new[0], p2_new[1]])

    return np.array(merged_lines, dtype=int)

def main():
    if os.path.exists(PCD_FILE_PATH):
        print(f"Loading point cloud from: {PCD_FILE_PATH}")
        pcd = o3d.io.read_point_cloud(PCD_FILE_PATH)
    else:
        pcd = create_placeholder_pcd()

    if not pcd.has_points():
        print("Error: Point cloud is empty. Cannot proceed.")
        return

    points_3d = np.asarray(pcd.points)

    print(f"Slicing point cloud at height {SLICE_HEIGHT}m with thickness {SLICE_THICKNESS}m...")
    slice_points_3d = points_3d[
        (points_3d[:, 2] >= SLICE_HEIGHT - SLICE_THICKNESS / 2) &
        (points_3d[:, 2] <= SLICE_HEIGHT + SLICE_THICKNESS / 2)
    ]
    points_2d = slice_points_3d[:, :2] 
    print(f"Found {points_2d.shape[0]} points in the slice.")

    print(f"Creating occupancy grid with {GRID_RESOLUTION*100}cm resolution...")
    occupancy_grid, grid_origin, grid_res = create_occupancy_grid(points_2d, GRID_RESOLUTION)
    if occupancy_grid is None:
        return

    print("Detecting line segments...")
    lsd = cv2.createLineSegmentDetector(0)
    lines_raw, _, _, _ = lsd.detect(occupancy_grid)
    
    if lines_raw is None:
        print("No lines detected. The occupancy grid might be too sparse.")
        lines_raw = []
    else:
        lines_raw = lines_raw.reshape(-1, 4)
        lengths = np.linalg.norm(lines_raw[:, 2:4] - lines_raw[:, 0:2], axis=1)
        lines_raw = lines_raw[lengths > LSD_MIN_LINE_LENGTH]
    print(f"Detected {len(lines_raw)} raw line segments.")

    print("Merging collinear lines...")
    lines_merged = merge_lines(lines_raw, MERGE_ANGLE_TOLERANCE, MERGE_DISTANCE_TOLERANCE)
    print(f"Refined to {len(lines_merged)} final lines.")


    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Floor Plan Generation Pipeline", fontsize=16)

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.scatter(points_2d[:, 0], points_2d[:, 1], s=1, c='black')
    ax1.set_title("1. 2D Point Cloud Slice")
    ax1.set_xlabel("X (meters)")
    ax1.set_ylabel("Y (meters)")
    ax1.axis('equal')
    ax1.grid(True)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(occupancy_grid, cmap='gray', origin='lower')
    ax2.set_title("2. Rasterized Occupancy Grid")
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(occupancy_grid, cmap='gray', origin='lower')
    for line in lines_raw:
        x1, y1, x2, y2 = line
        ax3.plot([x1, x2], [y1, y2], color='cyan', linewidth=1.5)
    ax3.set_title(f"3. Detected Raw Lines ({len(lines_raw)})")
    ax3.set_xticks([])
    ax3.set_yticks([])

    ax4 = fig.add_subplot(2, 2, 4)
    if len(lines_merged) > 0:
        for line in lines_merged:
            x1, y1, x2, y2 = line
            wx1, wy1 = np.array([x1, y1]) * grid_res + grid_origin
            wx2, wy2 = np.array([x2, y2]) * grid_res + grid_origin
            ax4.plot([wx1, wx2], [wy1, wy2], color='red', linewidth=2)
            
    ax4.set_title(f"4. Final Merged Vector Floor Plan ({len(lines_merged)})")
    ax4.set_xlabel("X (meters)")
    ax4.set_ylabel("Y (meters)")
    ax4.axis('equal')
    ax4.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    main()