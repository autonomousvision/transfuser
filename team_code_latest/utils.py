import numpy as np

def get_virtual_lidar_to_vehicle_transform():
    # This is a fake lidar coordinate
    T = np.eye(4)
    T[0, 3] = 1.3
    T[1, 3] = 0.0
    T[2, 3] = 2.5
    return T
        
def get_vehicle_to_virtual_lidar_transform():
    return np.linalg.inv(get_virtual_lidar_to_vehicle_transform())

def get_lidar_to_vehicle_transform():
    rot = np.array([[0, 1, 0],
                    [-1, 0, 0],
                    [0, 0, 1]], dtype=np.float32)
    T = np.eye(4)
    T[:3, :3] = rot

    T[0, 3] = 1.3
    T[1, 3] = 0.0
    T[2, 3] = 2.5
    return T

def get_vehicle_to_lidar_transform():
    return np.linalg.inv(get_lidar_to_vehicle_transform())

def get_lidar_to_bevimage_transform():
    # rot 
    T = np.array([[0, -1, 16],
                  [-1, 0, 32],
                  [0, 0, 1]], dtype=np.float32)
    # scale 
    T[:2, :] *= 8

    return T

def normalize_angle(x):
    x = x % (2 * np.pi)    # force in range [0, 2 pi)
    if x > np.pi:          # move to [-pi, pi)
        x -= 2 * np.pi
    return x

def normalize_angle_degree(x):
    x = x % 360.0
    if (x > 180.0):
        x -= 360.0
    return x