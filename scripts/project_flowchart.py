import graphviz

def create_pipeline_flowchart():
    dot = graphviz.Digraph('Pipeline', comment='Point Cloud to Floor Plan Pipeline')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.8', ranksep='1.2')
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='#E6F7FF', fontname='Helvetica')
    dot.attr('edge', fontname='Helvetica', fontsize='10')

    process_style = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#D6EAF8'}
    data_style = {'shape': 'cylinder', 'style': 'filled', 'fillcolor': '#E8F6F3'}
    io_style = {'shape': 'parallelogram', 'style': 'filled', 'fillcolor': '#FDF2E9'}
    sub_pipeline_style = {'style': 'filled', 'color': '#F4F6F7', 'labeljust': 'l'}

    # 1. Input
    dot.node('A_ply_file', 'Input .ply File', **io_style)

    # 2. Reading
    dot.node('B_read_pcd', '1. Read PLY to PCD\n(.ply to open3d format)', **process_style)

    # 3. Segmentation
    dot.node('C_segmentation', '2. 3D Semantic Segmentation\n(Pre-trained RandLANet on S3DIS)', **process_style)
    dot.node('C_data_segmented', 'Segmented PCD\n(with S3DIS labels)', **data_style)

    # 4. Cleaning
    dot.node('D_isolate', '3. Remove Unwanted Classes \n(Remove point clouds of chairs, sofas, tables, etc.)', **process_style)
    dot.node('D_data_structural', 'Structural PCD\n(Walls, Floor, Doors, etc.)', **data_style)
    dot.node('D_data_clutter', 'Clutter PCD\n(Clutter class of S3DIS from RandLANet)', **data_style)

    # 5. RANSAC Refinement
    dot.node('E_ransac', '4. Iterative RANSAC Refinement', **process_style)
    dot.node('E_data_extracted', 'Extracted Planar/Cylindrical Points', **data_style)
    
    # 6. Assembly
    dot.node('F_assembly', '5. PCD Assembly', **process_style)
    dot.node('F_data_final_pcd', 'Final Cleaned Structural PCD', **data_style)


    with dot.subgraph(name='cluster_floorplan') as sub:
        sub.attr(label='6. PCD to Floor Plan Pipeline', **sub_pipeline_style, fontname='Helvetica', fontsize='14')
        
        sub.node('G1_slice', 'Slice Point Cloud at Height', **process_style)
        sub.node('G2_grid', 'Create 2D Occupancy Grid', **process_style)
        sub.node('G3_clean', 'Clean Grid (Morphological Closing)', **process_style)
        sub.node('G4_hough', 'Detect Lines (Hough Transform)', **process_style)
        sub.node('G5_merge', 'Merge Collinear Lines', **process_style)
        sub.node('G6_connect', 'Connect Corners (Snap Endpoints)', **process_style)
        sub.node('G7_doors', 'Find Door Gaps\n(Compare Vector Plan to Grid)', **process_style)
        sub.node('G8_draw', 'Draw Final Plan with Doors', **process_style)

        sub.edge('G1_slice', 'G2_grid')
        sub.edge('G2_grid', 'G3_clean')
        sub.edge('G3_clean', 'G4_hough')
        sub.edge('G4_hough', 'G5_merge')
        sub.edge('G5_merge', 'G6_connect')
        sub.edge('G6_connect', 'G7_doors')
        sub.edge('G7_doors', 'G8_draw')

    dot.node('H_final_plan', 'High-Quality Floor Plan\n(.png file)', **io_style)


    dot.edge('A_ply_file', 'B_read_pcd')
    dot.edge('B_read_pcd', 'C_segmentation')
    dot.edge('C_segmentation', 'C_data_segmented')
    dot.edge('C_data_segmented', 'D_isolate')
    
    dot.edge('D_isolate', 'D_data_structural')
    dot.edge('D_isolate', 'D_data_clutter')
    
    dot.edge('D_data_clutter', 'E_ransac', label='Extract points from clutter')
    dot.edge('E_ransac', 'E_data_extracted')
    
    dot.edge('D_data_structural', 'F_assembly', label='Main structural elements')
    dot.edge('E_data_extracted', 'F_assembly', label='Refined elements')
    
    dot.edge('F_assembly', 'F_data_final_pcd')
    
    dot.edge('F_data_final_pcd', 'G1_slice', lhead='cluster_floorplan')

    dot.edge('G8_draw', 'H_final_plan', ltail='cluster_floorplan')

    output_filename = 'pipeline_flowchart'
    dot.render(output_filename, format='png', view=False, cleanup=True)
    print(f"Flowchart saved as '{output_filename}.png'")


if __name__ == '__main__':
    create_pipeline_flowchart()