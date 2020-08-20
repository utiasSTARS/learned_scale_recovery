import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.io as sio
import sys
sys.path.append('../')
from liegroups import SO3


def img_to_3d_torch(img, depth, K, ground_plane = None, depth_threshold=60, u_start=0, v_start=0):
    '''
    Parameters
    ----------
    img : pytorch image
        H x W x C
    depth : 
        H x W
    K : 3 x 3 intrinsic matrix

    Returns
    -------
    p : 3D coordinates for corresponding pixels on (u_grid, v_grid) - pixels with depths larger than depth_threshold are removed.

    '''
    f_x, f_y, c_x, c_y = K[0,0], K[1,1], K[0,2], K[1,2]
    img_samples = img 
    depth_samples = depth 

    u_grid = u_start + torch.arange(0,img_samples.size(1)).view((1,-1))
    v_grid = v_start + torch.arange(0,img_samples.size(0)).view((-1,1))

    u_grid = u_grid.repeat(img_samples.size(0),1).type_as(depth_samples)
    v_grid = v_grid.repeat(1,img_samples.size(1)).type_as(depth_samples)

    x = (depth_samples*(u_grid-c_x)/f_x).reshape((-1,1))
    y = (depth_samples*(v_grid-c_y)/f_y).reshape((-1,1))
    z = depth_samples.reshape((-1,1))
    p = torch.cat((x,y,z),dim=1)

    u_grid = u_grid.reshape((-1))
    v_grid = v_grid.reshape((-1))    
    idx= torch.where(p[:,2]<depth_threshold)[0]
    p = p[idx]
    u_grid = u_grid[idx]
    v_grid = v_grid[idx]
    
    return p, (u_grid, v_grid)

def fit_plane_torch(p, pixel_grid, ransac_iter=50, inlier_thresh=0.03):
    '''
    Parameters
    ----------
    p : array of 3D coordinates
    pixel_grid: (u,v) coordinates for p

    Returns
    -------
    best_normal : The normal direction of the plane that was fit using RANSAC
    best_inlier_list : indices of 3D points that are inliers
    (best_u, best_v) : image coordinates of inlier pixels
    '''

    u_grid, v_grid = pixel_grid
    b = torch.ones((3,1))
    most_inlier = 0
    
    for n in range(0,ransac_iter):
        #sample 3 points
        idx = []
        A = torch.zeros((3,3))
        while len(idx)<= 2:
            num = np.random.randint(0, p.shape[0])
            if num not in idx:
                idx.append(num)
        for i in range(0,3):
            A[i,:] = p[idx[i]]
        A_inv = torch.pinverse(A)   
        normal = A_inv.mm(b)  
        n = normal/torch.norm(normal)
        
        ## check number of inliers
        d = torch.abs(p.mm(normal)-1)
        inlier_list = torch.where(d<inlier_thresh)[0]

        inlier_u_pixels = u_grid[inlier_list] 
        inlier_v_pixels = v_grid[inlier_list] 
        
        if len(inlier_list) > most_inlier and (n[1,0]>0.9):
            most_inlier = len(inlier_list)
            best_inlier_list = inlier_list
            best_normal = normal
            best_u = inlier_u_pixels
            best_v = inlier_v_pixels
    
    b = torch.ones((best_inlier_list.size(0),1))
    best_normal = torch.pinverse(p[best_inlier_list]).mm(b)    ##recompute normal with all points
    
    return best_normal, best_inlier_list , (best_u, best_v)            
 