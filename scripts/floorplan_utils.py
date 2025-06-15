import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import math

SLICE_HEIGHT = 1.5 # Height in meters for the cross-section (e.g., eye-level).
SLICE_THICKNESS = 0.2 # Thickness of the slice (e.g., 20cm) to ensure we capture walls.
GRID_RESOLUTION = 0.02 # Size of each pixel in the 2D grid in meters (2cm).

HOUGH_THRESHOLD = 40      # Minimum number of intersections to detect a line. Lower -> more lines.
HOUGH_MIN_LINE_LENGTH = 40 # Minimum line length in pixels.
HOUGH_MAX_LINE_GAP = 20   # Max gap in pixels between segments to be joined into one line.

MERGE_ANGLE_TOLERANCE = 5 
MERGE_DISTANCE_TOLERANCE = 25
SNAP_TOLERANCE = 30

def create_occupancy_grid(points_2d, resolution):
    if points_2d.shape[0] == 0:
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
    if not isinstance(lines, np.ndarray) or lines.size == 0:
        return np.array([])
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
                line_len_sq = np.dot(line_vec, line_vec)
                if line_len_sq == 0:
                    i += 1
                    continue
                d = np.linalg.norm(np.cross(line_vec, p_base_1 - p1)) / np.linalg.norm(line_vec)
                if d < dist_thresh:
                    current_points.extend(list(comp_line.reshape(2, 2)))
                    lines_rad.pop(i)
                else: i += 1
            else: i += 1
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

def connect_corners(lines, snap_threshold):
    if len(lines) == 0:
        return np.array([])
    endpoints = []
    for line in lines:
        endpoints.append(np.array([line[0], line[1]]))
        endpoints.append(np.array([line[2], line[3]]))
    snapped = [False] * len(endpoints)
    new_lines = np.copy(lines).astype(float)
    for i in range(len(endpoints)):
        if snapped[i]: continue
        nearby_indices = [i]
        for j in range(i + 1, len(endpoints)):
            if snapped[j]: continue
            if np.linalg.norm(endpoints[i] - endpoints[j]) < snap_threshold:
                nearby_indices.append(j)
        if len(nearby_indices) > 1:
            avg_point = np.mean([endpoints[k] for k in nearby_indices], axis=0)
            for index in nearby_indices:
                snapped[index] = True
                line_index, point_index = divmod(index, 2)
                new_lines[line_index, point_index*2:point_index*2+2] = avg_point
    return new_lines.astype(int)

def pcd_to_plan(pcd, out_path):
    if not pcd.has_points():
        print("Error: Point cloud is empty. Cannot proceed.")
        return

    points_3d = np.asarray(pcd.points)
    slice_points_3d = points_3d[
        (points_3d[:, 2] >= SLICE_HEIGHT - SLICE_THICKNESS / 2) &
        (points_3d[:, 2] <= SLICE_HEIGHT + SLICE_THICKNESS / 2)
    ]
    points_2d = slice_points_3d[:, :2]

    print("Creating and cleaning occupancy grid...")
    grid_cleaned, grid_origin, grid_res = create_occupancy_grid(points_2d, GRID_RESOLUTION)
    if grid_cleaned is None: return

    print("Detecting lines with Hough Transform...")
    lines_raw = cv2.HoughLinesP(
        grid_cleaned, 
        rho=1, 
        theta=np.pi / 180, 
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP
    )
    
    if lines_raw is None:
        lines_raw = np.array([])
    else:
        lines_raw = lines_raw.reshape(-1, 4)
    print(f"Detected {len(lines_raw)} raw line segments with Hough.")

    print("Merging collinear lines...")
    lines_merged = merge_lines(lines_raw, MERGE_ANGLE_TOLERANCE, MERGE_DISTANCE_TOLERANCE)
    print(f"Refined to {len(lines_merged)} merged lines.")

    print("Snapping corners...")
    lines_connected = connect_corners(lines_merged, SNAP_TOLERANCE)
    print(f"Final plan has {len(lines_connected)} connected lines.")

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Floor Plan Generation Pipeline (Hough Transform Method)", fontsize=16)

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.scatter(points_2d[:, 0], points_2d[:, 1], s=1, c='black')
    ax1.set_title("1. 2D Point Cloud Slice")
    ax1.axis('equal'); ax1.grid(True)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(grid_cleaned, cmap='gray', origin='lower')
    ax2.set_title("2. Cleaned Occupancy Grid (Input for Hough)")

    ax3 = fig.add_subplot(2, 2, 3)
    grid_display = cv2.cvtColor(grid_cleaned, cv2.COLOR_GRAY2BGR)
    for line in lines_raw:
        x1, y1, x2, y2 = line
        cv2.line(grid_display, (x1, y1), (x2, y2), (0, 255, 255), 1) # Cyan lines
    ax3.imshow(grid_display, origin='lower')
    ax3.set_title(f"3. Detected Raw Hough Lines ({len(lines_raw)})")

    ax4 = fig.add_subplot(2, 2, 4)
    if len(lines_connected) > 0:
        for line in lines_connected:
            wx1, wy1 = np.array([line[0], line[1]]) * grid_res + grid_origin
            wx2, wy2 = np.array([line[2], line[3]]) * grid_res + grid_origin
            ax4.plot([wx1, wx2], [wy1, wy2], color='red', linewidth=2)
    ax4.set_title(f"4. Final Connected Vector Plan ({len(lines_connected)})")
    ax4.axis('equal'); ax4.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    fig.savefig(os.path.join(out_path, "floor_plan.png"), dpi=fig.dpi)