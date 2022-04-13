# Written by Amnon Drory (amnondrory@mail.tau.ac.il), Tel-Aviv University, 2021.
# Please cite the following paper if you use this code:
# - Amnon Drory, Shai Avidan, Raja Giryes, Stress Testing LiDAR Registration

import numpy as np
import math

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

def rotation_matrix_euler_angles(R, deg_or_rads='deg') :

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


def motion_to_fields(Homogenous):
    """
    Convert a 4x4 homogenous matrix to a vector of 
    [trans_x, trans_y, trans_z, roll, pitch, yaw]
    """
    T = Homogenous[:3,3]
    R = Homogenous[:3,:3]
    angles = rotation_matrix_euler_angles(R)
    vec = np.hstack([T.flatten(), angles.flatten()])
    return vec


def apply_transformation(M, X):
    X1 = np.hstack([X,np.ones_like(X)[:,0:1]])
    Y1_T = M @ X1.T
    return Y1_T.T[:,:3]
