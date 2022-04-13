import open3d as o3d
from copy import deepcopy
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

def draw_PC_with_normals(A,A_normals):
    A = c(A)
    A_normals = c(A_normals)
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(A)
    source.normals = o3d.utility.Vector3dVector(A_normals)
    o3d.visualization.draw_geometries([source])

def draw_PC(A, color_ind=0, title=None):
    if title is None:
        title = ''
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(A)
    source_temp = deepcopy(source)

    if color_ind==0:
        source_temp.paint_uniform_color([0, 0.706, 1])
    else:
        source_temp.paint_uniform_color([1, 0.706, 0])
    o3d.visualization.draw_geometries([source_temp],window_name=title)

def draw_normals(points, normals, ax, color, symmetric=False):
    if symmetric:
        line_segment_starts = points - normals
    else:
        line_segment_starts = points

    line_segment_ends = points + normals

    line_segments = np.dstack([line_segment_starts, line_segment_ends])

    for i in range(line_segments.shape[0]):
        ax.plot(xs=line_segments[i,0,:], ys=line_segments[i,1,:], zs=line_segments[i,2,:], c=color.reshape([1,3]))

def visualize_normal_prediction(full_PC, initial_PC, GT_normals, pred_normals):
    if torch.is_tensor(initial_PC):
        initial_PC = c(initial_PC)
    if torch.is_tensor(GT_normals):
        GT_normals = c(GT_normals)
    if torch.is_tensor(pred_normals):
        pred_normals = c(pred_normals)

    final_PC = initial_PC[:GT_normals.shape[0],:]

    # fig = plt.gcf()
    # ax = fig.add_subplot(111, projection='3d')
    # emphasis = np.zeros([initial_PC.shape[0]], dtype=bool)
    # emphasis[:GT_normals.shape[0]] = True
    # display_with_emphasis(initial_PC, emphasis,ax, [0,0,1])
    # draw_normals(final_PC, GT_normals, ax, [0,1,0])
    # plt.show()

    def calc_normal_diff(nA,nB):
        nA_ = nA / np.sum(nA,axis=1,keepdims=True)
        nB_ = nB / np.sum(nB, axis=1,keepdims=True)
        raw_angle = np.arccos(np.sum(nA_ * nB_,axis=1))
        angle = np.minimum(np.pi - raw_angle, raw_angle)
        return angle

    normal_diff = calc_normal_diff(GT_normals,pred_normals)
    threshes = np.linspace(0,np.pi/2,10)
    print(threshes)
    print(np.rad2deg(threshes))
    labels = np.zeros(normal_diff.shape[0], dtype=np.long)
    for i, t in enumerate(threshes):
        is_higher = normal_diff >= t
        labels[is_higher] = i

    SHRINK_FACTOR = 0.1
    A_normals = SHRINK_FACTOR * GT_normals
    B_normals = SHRINK_FACTOR * pred_normals


    cmap = cm.get_cmap('jet', np.max(labels)+1)
    colors = []
    for i in range(np.max(labels)+1):
        colors.append(cmap(i)[:3])
    colors = np.vstack(colors)

    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(final_PC)
    source.normals = o3d.utility.Vector3dVector(A_normals)
    source.colors = o3d.utility.Vector3dVector(colors[labels])
    inv_source = o3d.geometry.PointCloud()
    inv_source.points = o3d.utility.Vector3dVector(final_PC)
    inv_source.normals = o3d.utility.Vector3dVector(-A_normals)
    inv_source.colors = o3d.utility.Vector3dVector(colors[labels])

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(final_PC)
    target.normals = o3d.utility.Vector3dVector(B_normals)
    target.colors = o3d.utility.Vector3dVector(colors[labels])
    inv_target = o3d.geometry.PointCloud()
    inv_target.points = o3d.utility.Vector3dVector(final_PC)
    inv_target.normals = o3d.utility.Vector3dVector(-B_normals)
    inv_target.colors = o3d.utility.Vector3dVector(colors[labels])

    Back = o3d.geometry.PointCloud()
    Back.points = o3d.utility.Vector3dVector(initial_PC[final_PC.shape[0]:])
    Back.paint_uniform_color([0.7, 0.7, 0.7])

    DeepBack = o3d.geometry.PointCloud()
    DeepBack.points = o3d.utility.Vector3dVector(full_PC)
    DeepBack.paint_uniform_color([0.85, 0.85, 0.85])

    #o3d.visualization.draw_geometries([DeepBack, Back, source, inv_source, target, inv_target])
    #o3d.visualization.draw_geometries([Back, source, inv_source, target, inv_target])
    o3d.visualization.draw_geometries([source, inv_source, target, inv_target])




    
    

def draw_PC_with_labels(A, labels):
    A = c(A)
    labels = c(labels)
    labels = labels.astype(np.long).flatten()
    cmap = cm.get_cmap('Dark2', np.max(labels)+1)
    colors = cmap.colors[:,:3]
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(A)
    source.colors = o3d.utility.Vector3dVector(colors[labels])
    o3d.visualization.draw_geometries([source])

def draw_pairs(A,B,A_i,B_j):
    colors = [[1, 0.706, 0],  [0, 0.651, 0.929], [0.8, 0.8, 0.8]]
    NUM_COLORS = len(colors)
    sources = []

    PC_A = c(A[:,:3])
    source_A = o3d.geometry.PointCloud()
    source_A.points = o3d.utility.Vector3dVector(PC_A)
    source_A.paint_uniform_color(colors[0])

    PC_B = c(B[:,:3])
    source_B = o3d.geometry.PointCloud()
    source_B.points = o3d.utility.Vector3dVector(PC_B)
    source_B.paint_uniform_color(colors[1])        

    points = np.vstack([c(A_i), c(B_j)])
    lines = []
    line_colors = []
    NUM_LINE_COLORS = NUM_COLORS - 2
    x_min = np.min(points[:,0])
    x_max = np.max(points[:,0])
    x_range = x_max-x_min
    x_step = x_range / NUM_LINE_COLORS

    for i in range(A_i.shape[0]):
        lines.append([i, i+A_i.shape[0]])
        color_ind = int(np.floor((points[i,0]-x_min)/x_step))
        line_colors.append(colors[2+color_ind])
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(line_colors)
    o3d.visualization.draw_geometries([source_A, source_B, line_set])

def draw_multiple_clouds(*args):
    colors = [[1, 0.706, 0],  [0, 0.651, 0.929], [0,1,0], [0.8, 0.8, 0.8], [0,0,0], [1,0,0],[0,0,1]]
    sources = []

    for i,A in enumerate(args):
        A = c(A)
        PC = A[:,:3]
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(PC)
        source.paint_uniform_color(colors[i])
        sources.append(source)
    o3d.visualization.draw_geometries(sources)

def draw_registration_result(A,B,title=None):
    if title is None:
        title = ''
    PC_1 = A[:,:3]
    PC_2 = B[:,:3]
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(PC_1)
    target.points = o3d.utility.Vector3dVector(PC_2)
    source_temp = deepcopy(source)
    target_temp = deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    #source_temp.paint_uniform_color([0.8, 0.8, 0.8])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    #target_temp.paint_uniform_color([1, 0.706, 0])
    o3d.visualization.draw_geometries([source_temp, target_temp],window_name=title)

def visualize_gaussian(PC, feature):
    NUM_SAMPLES = 500
    mean = feature[:3]
    inv_cov = np.array(feature[3:]).reshape([3,3])
    cov = np.linalg.inv(inv_cov)
    samples = np.random.multivariate_normal(mean, cov, NUM_SAMPLES)
    draw_registration_result(PC, samples)

def DisplayPoints(A, B=None, A_emphasis=None, B_emphasis=None):
    fig = plt.gcf()
    ax = fig.add_subplot(111, projection='3d')
    #ax.set_aspect('equal')

    display_with_emphasis(A, A_emphasis, ax, [0, 0, 1])
    display_with_emphasis(B, B_emphasis, ax, [1, 0, 0])

    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    set_axes_equal(ax)

def display_with_emphasis(PC, emphasis, ax, color):
    BACKGROUND_WASHOUT = 0.9
    if PC is None:
        return
    if emphasis is None:
        emphasis = np.ones(PC.shape[0], dtype=np.bool)

    front = PC[emphasis, ...]
    background = PC[~emphasis, ...]
    ax.scatter(front[:, 0], front[:, 1], front[:, 2], c=color)
    # background_color = (1-BACKGROUND_WASHOUT)*np.array(color) + BACKGROUND_WASHOUT*np.array([1.,1.,1.])
    background_color = [list(color) + [(1 - BACKGROUND_WASHOUT)]]
    ax.scatter(background[:, 0], background[:, 1], background[:, 2], c=background_color)


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def get_second_degree_neighborhood(ind, nei_idx):
    first_deg = nei_idx[ind, :]
    sec_deg_list = []
    for i in first_deg:
        sec_deg_list.append(nei_idx[i, :])
    sec_deg = np.vstack(sec_deg_list).flatten()
    sec_deg = np.array(list(set(sec_deg) - set(first_deg)))
    all_inds = np.array(list(set(np.arange(nei_idx.shape[0])) - set(sec_deg) - set(first_deg)))
    return all_inds, first_deg, sec_deg

def c(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    else:
        return x

def visualize_neighbors_in_subclouds(PC, neighbors, RANDLA_N_POINTS_OUT, ind_in_batch=0, pixel_ind=0):

    if torch.is_tensor(PC):
        PC = c(PC)
        new_neighbors = [[c(t) for t in s] for s in neighbors]
        neighbors = new_neighbors

    if len(PC.shape) == 3:
        PC = PC[ind_in_batch,...]
    if len(neighbors[0][0].shape) == 3:
        new_neighbors = [[t[ind_in_batch,...] for t in s] for s in neighbors]
        neighbors = new_neighbors

    num_layers = len(neighbors)

    N = PC.shape[0]
    print(N)

    all_inds, first_deg, sec_deg = get_second_degree_neighborhood(pixel_ind, neighbors[0][0])
    A = PC[all_inds, :]
    B = PC[first_deg, :]
    C = PC[sec_deg, :]
    draw_multiple_clouds(A, B, C)
    for i in range(num_layers-1):
            N = RANDLA_N_POINTS_OUT[i]
            PC = PC[:N, :]
            all_inds, first_deg, sec_deg = get_second_degree_neighborhood(pixel_ind, neighbors[i + 1][0])
            A = PC[all_inds, :]
            B = PC[first_deg, :]
            C = PC[sec_deg, :]
            print(N)
            draw_multiple_clouds(A, B, C)

def compare_PC_and_normals(A, A_normals, B, B_normals):
    if torch.is_tensor(A):
        A = c(A)
        A_normals = c(A_normals)
        B = c(B)
        B_normals = c(B_normals)
    assert len(A.shape) == 2
    assert len(A_normals.shape) == 2
    assert len(B.shape) == 2
    assert len(B_normals.shape) == 2
    assert (A.shape == A_normals.shape)
    assert (B.shape == B_normals.shape)

    B_moved = B + np.max(A,axis=0,keepdims=True) - np.min(A,axis=0,keepdims=True)
    C = np.vstack([A,B_moved])
    C_normals = np.vstack([A_normals,B_normals])
    draw_PC_with_normals(C, C_normals)

def visualize_clouds(arrs):
    SHOW_NORMALS = False
    SHOW_NEIGHBORS = True
    SHOW_LABELS = False
    neighbors = arrs[0]['neighbors']
    PC = arrs[0]['PC']
    normals = arrs[0]['normals']
    labels = arrs[0]['labels']

    decimation = 4
    num_layers = 4

    if SHOW_NORMALS:
        N = PC.shape[0]
        print(N)
        draw_PC_with_normals(PC, normals)
        for i in range(num_layers):
            N = N // decimation
            print(N)
            PC = PC[:N, :]
            normals = normals[:N, :]
            draw_PC_with_normals(PC, normals)

    if SHOW_NEIGHBORS:
        visualize_neighbors_in_subclouds(PC, neighbors)

    if SHOW_LABELS:
        N = PC.shape[0]
        print(N)
        draw_PC_with_labels(PC, labels)
        for i in range(num_layers):
            N = N // decimation
            print(N)
            PC = PC[:N, :]
            labels = labels[:N]
            draw_PC_with_labels(PC, labels)