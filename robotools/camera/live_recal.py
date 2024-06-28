import cv2
from cv2 import aruco
import open3d as o3d
import numpy as np
import robotools as rt
from robotools.camera import Realsense, HandeyeCalibrator, zed
from robotools.geometry import (
    invert_homogeneous
)




def charucoPnP(calibrator,cam, img,world2cam_raw, world2marker):
    calibrator.reset()
    vis = calibrator.capture(img, world2cam_raw)

    #find charucomarkes
    detection_result = calibrator._detect_charuco(img, world2cam_raw)
    if detection_result.detected == False:
        return False, None
   
    #split cam pose into two vectors
    
    
    #possible usage of intial guesses, doesnt work
    #rvec_guess, _ = cv2.Rodrigues(cam2marker_raw[:3, :3])   
    #tvec_guess =  cam2marker_raw[:3, 3]


    #estimatePoseCharucoBoard
    if detection_result.detected:

        retval, rvec_out, tvec_out = cv2.aruco.estimatePoseCharucoBoard(
            charucoCorners = detection_result.corners,
            charucoIds = detection_result.ids,
            board = calibrator.charuco_board,
            cameraMatrix = cam.calibration.intrinsic_matrix,
            distCoeffs = cam.calibration.dist_coeffs,
            rvec = None,
            tvec = None,
            useExtrinsicGuess = False
        )
        if retval:
            #fuse estimated camera position
            rmat, _ = cv2.Rodrigues(rvec_out)
            cam2marker = np.eye(4)
            cam2marker[:3, :3] =rmat
            cam2marker[:3, 3] = tvec_out.ravel()
            marker2cam = invert_homogeneous(cam2marker)
        
            #multiply estimated camera postion and W2m
            world2cam_refined =  world2marker @ marker2cam
            retval = True 
            return retval, world2cam_refined
        else:
            print("Pose estimation failed")
    retval = False
    world2cam_refined = None
    return retval, world2cam_refined