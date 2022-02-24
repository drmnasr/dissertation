#
#
# Credit for https://github.com/jdgalviss/autonomous_mobile_robot for these 3 functions
#
#
import numpy as np
import cv2

def get_label_colors(driveable = True):
    if(driveable >= 0):
        colors = [  [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]
                ]
        colors[driveable] = [55, 195, 55]
    else:
        colors = [
                [0, 0, 0],
                [128, 64, 128],
                [55, 195, 55],
                [244, 35, 232],
                [70, 70, 70],
                [190, 153, 153],
                [153, 153, 153],
                [250, 170, 30],
                [220, 220, 0],
                [107, 142, 35],
                [152, 251, 152],
                [0, 130, 180],
                [220, 20, 60],
                [255, 0, 0],
                [0, 0, 142],
                [0, 0, 70],
                [0, 60, 100],
                [0, 80, 100],
                [0, 0, 230],
                [119, 11, 32],
                [0, 0, 192]
            ]
        
    return dict(zip(range(21), colors))


def colorize(labels, label_colors):
        labels = np.clip(labels, 0, len(label_colors)-1)
        r = labels.copy()
        g = labels.copy()
        b = labels.copy()
        for l in range(0, 21):
            r[labels == l] = label_colors [l][0]
            g[labels == l] = label_colors [l][1]
            b[labels == l] = label_colors [l][2]
        rgb = np.zeros((labels.shape[0], labels.shape[1], 3))
        rgb[:, :, 0] = r 
        rgb[:, :, 1] = g 
        rgb[:, :, 2] = b 
        return np.uint8(rgb)

def warp(mask):
    mtxs = np.load('./warp.npy')
    M_ = mtxs
    M_inv_ = np.linalg.inv(mtxs)
    h,w,_ = mask.shape
    # Warp driveable area
    warped = cv2.warpPerspective(mask, M_, (960, 720), flags=cv2.INTER_LINEAR)
    
    # Calculate robot center
    original_center = np.array([[[w/2,h]]],dtype=np.float32)
    warped_center = cv2.perspectiveTransform(original_center, M_)[0][0]
    return warped, warped_center, original_center