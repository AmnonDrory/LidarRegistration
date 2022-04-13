# Written by Amnon Drory (amnondrory@mail.tau.ac.il), Tel-Aviv University, 2021.
# Please cite the following paper if you use this code:
# - Amnon Drory, Shai Avidan, Raja Giryes, Stress Testing LiDAR Registration

import open3d as o3d

def draw_multiple_clouds(*args):
    colors = [[1, 0.706, 0],  [0, 0.651, 0.929], [0,1,0], [0.8, 0.8, 0.8], [0,0,0], [1,0,0],[0,0,1]]
    sources = []

    for i,A in enumerate(args):
        PC = A[:,:3]
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(PC)
        source.paint_uniform_color(colors[i])
        sources.append(source)
    o3d.visualization.draw_geometries(sources)

