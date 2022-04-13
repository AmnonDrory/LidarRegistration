import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
import torch
from torch.autograd import Variable
import random
import open3d as o3d
from utils.subsampling import num_features

def euler_to_quaternion(yaw, pitch, roll, deg_or_rad='rad'):
        if deg_or_rad=='deg':
            yaw = np.deg2rad(yaw)
            pitch = np.deg2rad(pitch)
            roll = np.deg2rad(roll)

        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

        return [qx, qy, qz, qw]

def quaternion_to_euler(x, y, z, w):
        'Euler angles output are in deg'
        import math
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        X = math.degrees(math.atan2(t0, t1))

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        Y = math.degrees(math.asin(t2))

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        Z = math.degrees(math.atan2(t3, t4))

        return X, Y, Z


def calc_translational_error(GT_mot, res_mot):
    err = np.linalg.norm(GT_mot['trans'].flatten() - res_mot['trans'].flatten())
    return err

def calc_angular_error(vec1, vec2):
    R1 = euler_angles_to_rotation_matrix(vec1)
    R2 = euler_angles_to_rotation_matrix(vec2)
    norm = Frobenius_Norm(R1 - R2)
    theta = np.rad2deg( 2 * np.arcsin(norm/np.sqrt(8)) )
    return theta

def calc_angular_error_FCGF(angles_GT, angles):
    """
    Calcualte angular error as defined in the FCGF paper
    """
    R_GT = euler_angles_to_rotation_matrix(angles_GT)
    R = euler_angles_to_rotation_matrix(angles)
    return np.rad2deg(np.arccos((np.trace(np.matmul(R.T, R_GT))-1)/2))

def Frobenius_Norm(R):
    norm = np.sqrt(np.trace(np.matmul(R, R.T)))
    return norm

def sample_pc(pc, n_samples):
    # downsample
    pcd_A = o3d.geometry.PointCloud()
    pcd_A.points = o3d.utility.Vector3dVector(pc)
    downsampled_A = voxel_filter(pcd_A, n_samples)
    A = pad_with_samples(downsampled_A, pcd_A, n_samples)
    return A

def pad_with_samples(pcd, pcd_samples, N):
    # pad pcd with samples from pcd_samples to the size N.
    assert(np.shape(pcd.points)[1] is 3)
    assert(np.shape(pcd_samples.points)[1] is 3)
    K = np.shape(pcd.points)[0]
    samples =  np.random.choice(np.shape(pcd_samples.points)[0], N-K)
    pcd_xyz = np.asarray(pcd.points)
    pcd_samples_xyz = np.asarray(pcd_samples.points)
    padded = np.concatenate((pcd_xyz, pcd_samples_xyz[samples]))
    return padded

def euclidian_distance(x1, x2):
    '''3D Euclidian distance'''
    return np.linalg.norm(x1-x2)

def create_dist_mat(pc1, pc2):
    col = 0
    row = 0
    assert(pc1.shape[1] is 3 and pc2.shape[1] is 3)
    dist_mat = np.full((pc1.shape[0], pc2.shape[0]), np.nan)
    for pcx1 in pc1:
        for pcx2 in pc2:
            dist_mat[row, col] = euclidian_distance(pcx1, pcx2)
            col+=1
        col=0
        row+=1
    return dist_mat

def plot_3d(coords, figure=1, show=True):
    markers = ['o','^','*','-']
    colors = ['r','b','g','y']
    if np.shape(coords.shape)[0] < 3:
        N = 1
        coords = np.expand_dims(coords, axis=0)
    else:
        N = coords.shape[0]
    fig = plt.figure()
    n = 1
    ax = fig.add_subplot(111,projection='3d')
    for n in range(N):
        x = coords[n, 0, :]
        y = coords[n, 1, :]
        z = coords[n, 2, :]
        ax.scatter(x, y, z, marker=markers[n], c=colors[n])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    if show:
        plt.show()
    return fig, ax

def plot_3d_torch(coords, ax):
    markers = ['o','^','*','^']
    colors = ['r','b','g','y']
    # if np.shape(coords.shape)[0] < 3:
    #     N = 1
    #     coords = np.expand_dims(coords, axis=0)
    # else:
    N = coords.shape[1]
    # fig = plt.figure()
    n = 1
    m = markers[random.randint(0,3)]
    c = colors[random.randint(0,3)]
    for n in range(N):
        x = coords[0, n].item()
        y = coords[1, n].item()
        z = coords[2, n].item()
        ax.scatter(x, y, z, marker=m, c=c)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    return

def apply_raw_motion_torch(X, mot):
    theta = torch.tensor([mot['angles'][0]], device=X.device, dtype=X.dtype)
    phi = torch.tensor([mot['angles'][1]], device=X.device, dtype=X.dtype)
    psi = torch.tensor([mot['angles'][2]], device=X.device, dtype=X.dtype)
    R = euler_angles_to_rotation_matrix_torch(theta, phi, psi, deg_or_rad='deg')
    T = torch.tensor(mot['trans'].reshape([1,-1]), device=X.device, dtype=torch.double)
    res = rotate_3d_torch(X, R) + T.double()
    return res

def apply_rot_trans_torch(X,R,T, feature_mode='xyz'):
    if feature_mode == 'xyz':
        res = rotate_3d_torch(X,R) + T.double()
    else:
        xyz = X[:3,:]
        res_xyz = apply_rot_trans_torch(xyz,R,T, 'xyz')
        if feature_mode == 'density':
            density = X[3,:].unsqueeze(0)
            res = torch.cat([res_xyz,density],dim=0)
        elif feature_mode == 'gaussian':
            inv_cov = X[3:,:].t().reshape([-1,3, 3])
            inv_cov_ = inv_cov.double()
            R_ = R.double()
            left2 = torch.matmul(R_, inv_cov_,)
            res_inv_cov = torch.matmul(left2, R_.t())
            res_inv_cov_flat = res_inv_cov.reshape([-1,9]).t()
            res = torch.cat([res_xyz, res_inv_cov_flat], dim=0)

    return res

def rotate_3d_torch(X, R):
    """
    :param X: a torch tensor of 3xN or Nx3
    :param R: a rotation torch tensor 3x3, or a torch vector of 3 euler angles
    :return:
    """
    if R.nelement() == 3:
        R = R.double().clone()
        R = euler_angles_to_rotation_matrix_torch(R[0:1], R[1:2], R[2:3], deg_or_rad='deg')
    assert len(R.shape) == 2 and (R.shape[0] == 3) and (
                R.shape[1] == 3), 'Error: input should be rotation matrix, but has a shape: ' + str(R.shape)
    trans_flag = False
    if X.shape[0] is not 3:
        X = X.t()
        trans_flag = True
    Y = torch.matmul(R.double(), X.double())
    if trans_flag:
        Y = Y.t()
    return Y


def rotate_3d_old(X, R):
    # assert(isRotationMatrix(R))
    'X is a matrix of 3xN'
    'R is a rotation matrix 3x3'
    # trans_flag = False
    if X.shape[0] is not 3:
        X = X.T
        # trans_flag = True
    Y = np.matmul(R, X)
    # if trans_flag:
        # Y = Y.T
    return Y

def rotate_3d(X, R):
    """

    :param X: a matrix of 3xN or Nx3
    :param R: a rotation matrix 3x3, or a vector of 3 euler angles
    :return:
    """
    R = np.array(R) # to handle lists
    if R.size == 3:
        R = euler_angles_to_rotation_matrix(R)
    assert len(R.shape)==2 and (R.shape[0] == 3) and (R.shape[1] == 3), 'Error: input should be rotation matrix, but has a shape: ' + str(R.shape)
    trans_flag = False
    if X.shape[0] is not 3:
        X = X.T
        trans_flag = True
    Y = np.matmul(R, X)
    if trans_flag:
        Y = Y.T
    return Y

def rotate_in_random_angle(S, range, axes='one'):
    if axes is 'one':
        angle = 0
        while angle is 0:
            angle = random.uniform(-range/2, range/2)
        R = euler_angles_to_rotation_matrix([0,0,angle])
    else:
        angleX= 0
        angleY= 0
        angleZ= 0
        while angleX is 0 or angleY is 0 or angleZ is 0:
            angleX = random.uniform(-range/2, range/2)
            angleY = random.uniform(-range/2, range/2)
            angleZ = random.uniform(-range/2, range/2)
        R = euler_angles_to_rotation_matrix([angleX, angleY, angleZ])
        angle = [angleX, angleY, angleZ]
    return rotate_3d_old(S, R), angle

def euler_angles_to_rotation_matrix_torch(theta, phi, psi, deg_or_rad='rad'):
    theta = theta.reshape([1])
    phi = phi.reshape([1])
    psi = psi.reshape([1])
    if deg_or_rad == 'deg':
        theta *= np.pi/180.0
        phi *= np.pi/180.0
        psi *= np.pi/180.0

    one = Variable(torch.ones(1, dtype=theta.dtype)).to(theta.device)
    zero = Variable(torch.zeros(1, dtype=theta.dtype)).to(theta.device)

    rot_x = torch.cat((
        torch.unsqueeze(torch.cat((one, zero, zero), 0), dim=1),
        torch.unsqueeze(torch.cat((zero, theta.cos(), theta.sin()), 0), dim=1),
        torch.unsqueeze(torch.cat((zero, -theta.sin(), theta.cos()), 0), dim=1),
    ), dim=1)
    rot_y = torch.cat((
        torch.unsqueeze(torch.cat((phi.cos(), zero, -phi.sin()), 0), dim=1),
        torch.unsqueeze(torch.cat((zero, one, zero), 0), dim=1),
        torch.unsqueeze(torch.cat((phi.sin(), zero, phi.cos()), 0), dim=1),
    ), dim=1)
    rot_z = torch.cat((
        torch.unsqueeze(torch.cat((psi.cos(), psi.sin(), zero), 0), dim=1),
        torch.unsqueeze(torch.cat((-psi.sin(), psi.cos(), zero), 0), dim=1),
        torch.unsqueeze(torch.cat((zero, zero, one), 0), dim=1),
    ), dim=1)
    A = torch.mm(rot_z, torch.mm(rot_y, rot_x))
    return A

def R_T_to_dict(R,T):
    res = {'trans': T.flatten(), 'angles':rotation_matrix_euler_angles(R, 'deg')}
    return res

def euler_angles_to_rotation_matrix(theta_vec, deg_or_rad='deg'):
    if deg_or_rad is 'deg':
        theta_vec = np.radians(theta_vec)
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta_vec[0]), -math.sin(theta_vec[0]) ],
                    [0,         math.sin(theta_vec[0]), math.cos(theta_vec[0])  ]
                    ])
    R_y = np.array([[math.cos(theta_vec[1]),    0,      math.sin(theta_vec[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta_vec[1]),   0,      math.cos(theta_vec[1])  ]
                    ])
    R_z = np.array([[math.cos(theta_vec[2]),    -math.sin(theta_vec[2]),    0],
                    [math.sin(theta_vec[2]),    math.cos(theta_vec[2]),     0],
                    [0,                     0,                      1]
                    ])
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R

def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotation_matrix_euler_angles_torch(R, deg_or_rads='deg'):
    R_np = R.detach().cpu().numpy()
    euler_np = rotation_matrix_euler_angles(R_np, deg_or_rads)
    euler = torch.tensor(euler_np, device=R.device)
    return euler

def rotation_matrix_euler_angles(R, deg_or_rads='deg') :

    # assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    res= np.array([x, y, z])
    if deg_or_rads=="deg":
        res = np.degrees(res)

    return res

def plot_A_B(A,B, show=True):
    A = np.squeeze(A)
    B = np.squeeze(B)
    if A.shape[0] is not 3:
        A = A.T
    if B.shape[0] is not 3:
        B = B.T
    coords = np.concatenate([np.expand_dims(A, axis=0), np.expand_dims(B, axis=0)])
    _,ax = plot_3d(coords, show=show)
    return ax

def pc_size(PC):
    if PC.shape[1] is not 3:
        PC = PC.T
    range = np.max(PC,axis=0) - np.min(PC,axis=0)
    return np.linalg.norm(range)

def apply_random_rot_trans(PC, rot_range, trans_range):
    vec = np.random.uniform(-rot_range,rot_range,3)
    R = euler_angles_to_rotation_matrix(vec)
    T = np.random.uniform(-trans_range,trans_range,3)
    return rotate_3d_old(PC, R).T + T.T, vec, T

def apply_rot(PC, vec):
    R = euler_angles_to_rotation_matrix(vec)
    return rotate_3d_old(PC, R)

def apply_rot_trans_old(PC, R, T):
    newPC = rotate_3d_old(PC, R) + T
    return newPC

def apply_rot_trans_take2(PC, R, T):
    T = np.reshape(T, [3])
    newPC = rotate_3d(PC, R) + T
    return newPC

def calc_inverse_motion_from_R_T_torch(R,T):
    T = T.reshape([-1,1])
    # y = Rx + T
    # R.T(y - T) = x
    # x = R.T*y + (-R.T*T)
    inverse_R = R.t()
    inverse_T = -torch.matmul(inverse_R,T)
    inverse_T = inverse_T.reshape(T.shape)
    return inverse_R, inverse_T

def calc_inverse_motion(mot):
    R = euler_angles_to_rotation_matrix(mot['angles'])
    T = mot['trans'].reshape([-1,1])
    # y = Rx + T
    # R.T(y - T) = x
    # x = R.T*y + (-R.T*T)
    inverse_R = R.T
    inverse_T = -np.matmul(inverse_R,T)
    inverse_mot = {}
    inverse_mot['angles'] = rotation_matrix_euler_angles(inverse_R)
    inverse_mot['trans'] = inverse_T.flatten()
    return inverse_mot

def apply_transformation(M, X):
    if torch.is_tensor(M):
        X1 = torch.cat([X,torch.ones_like(X)[:,0:1]], dim=1)
        Y1_T = M @ X1.T
        return Y1_T.T[:,:3]    
    else:
        X1 = np.hstack([X,np.ones_like(X)[:,0:1]])
        Y1_T = M @ X1.T
        return Y1_T.T[:,:3]    

def apply_rot_trans(X,R,T, feature_mode='xyz'):

    if np.array(R).size == 3:
        R = euler_angles_to_rotation_matrix(R)

    is_transposed = X.shape[0] != num_features[feature_mode]

    if is_transposed:
        X = X.T

    assert X.shape[0] == num_features[feature_mode], "Feature mode %s doesn't match shape of points array: " % feature_mode + str(X.shape)

    T = np.reshape(T, [3,1])
    if feature_mode == 'xyz':
        res = rotate_3d(X,R) + T
    else:
        xyz = X[:3,:]
        res_xyz = apply_rot_trans(xyz,R,T, 'xyz')
        if feature_mode == 'density':
            density = X[3,:]
            res = np.vstack([res_xyz,density])
        elif feature_mode == 'gaussian':
            inv_cov = X[3:,:].T.reshape([-1,3, 3])

            left2 = np.matmul(R, inv_cov)
            res_inv_cov = np.matmul(left2, R.T)
            res_inv_cov_flat = res_inv_cov.reshape([-1,9]).T
            res = np.vstack([res_xyz, res_inv_cov_flat])

    if is_transposed:
        res = res.T

    return res


def dilate_randomly(PC, K, axis=0):

    if K>=np.max((PC.shape[0], PC.shape[1])):
        return PC
    if axis is 0:
        if K is not 0:
            samples = random.sample(population=range(PC.shape[0]), k=K)
            return PC[samples]
        else:
            return []
    else:
        if K is not 0:
            samples = random.sample(population=range(PC.shape[1]), k=K)
            return PC[:,samples]
        else:
            return []

def from_quat_to_rot_matrix(angle, n, deg_or_rads='deg'):
    'angle - the rotation angle size'
    'n - the axis of rotation'
    if deg_or_rads is 'deg':
        angle = angle/180*np.pi
    e1 = n[0]
    e2 = n[1]
    e3 = n[2]
    A11 = (1-np.cos(angle))*np.power(e1,2)+np.cos(angle)
    A12 = (1-np.cos(angle))*e1*e2-e3*np.sin(angle)
    A13 = (1-np.cos(angle))*e1*e3+e2*np.sin(angle)
    A21 = (1-np.cos(angle))*e1*e2+e3*np.sin(angle)
    A22 = (1-np.cos(angle))*np.power(e2,2)+np.cos(angle)
    A23 = (1-np.cos(angle))*e2*e3-e1*np.sin(angle)
    A31 = (1-np.cos(angle))*e1*e3-e2*np.sin(angle)
    A32 = (1-np.cos(angle))*e3*e2+e1*np.sin(angle)
    A33 = (1-np.cos(angle))*np.power(e3,2)+np.cos(angle)
    R = np.asarray([[A11, A12, A13],[A21, A22, A23],[A31, A32, A33]])
    return np.squeeze(R)

def rotate_around_axis(PC, n, angle):
    was_transposed = False
    if PC.shape[0] is not 3:
        PC = PC.T
        was_transposed = True
    R = from_quat_to_rot_matrix(angle, n)
    newPC = rotate_3d_old(PC, R)
    if was_transposed:
        newPC = newPC.T
    return newPC, R

def rotate_around_axis_and_center_of_rotation(PC, center_point_of_rotation, axis_of_rotation, angle):

    input_was_transposed = False

    if PC.shape[0] is not 3:
        input_was_transposed = True
        PC = PC.T

    center_point_of_rotation = center_point_of_rotation.reshape([-1, 1])
    PC_centered = PC - center_point_of_rotation

    new_PC_centered, R = rotate_around_axis(PC_centered, axis_of_rotation, angle)

    newPC = new_PC_centered + center_point_of_rotation

    motion = {}
    motion['angles'] = rotation_matrix_euler_angles(R)
    motion['trans'] = np.array(-np.matmul(R,center_point_of_rotation) + center_point_of_rotation)

    if input_was_transposed:
        newPC = newPC.T
    return newPC, motion

def add_motions(mot1, mot2):
    R1 = euler_angles_to_rotation_matrix(mot1['angles'])
    t1 = mot1['trans']
    R2 = euler_angles_to_rotation_matrix(mot2['angles'])
    t2 = mot2['trans']
    R = np.matmul(R2,R1)
    t = np.matmul(R2,t1.reshape([-1,1])) + t2.reshape([-1,1])
    mot = {'trans': t, 'angles': rotation_matrix_euler_angles(R)}
    return mot

def homogeneous_to_dict(Homogenous):
    """
    Convert a 4x4 homogenous matrix to a vector of 
    [trans_x, trans_y, trans_z, roll, pitch, yaw]
    """
    T = Homogenous[:3,3]
    R = Homogenous[:3,:3]
    angles = rotation_matrix_euler_angles(R)
    mot = {'trans': T.flatten(), 'angles': angles}
    return mot
