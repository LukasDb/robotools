import cv2
import logging
from cv2 import aruco
from scipy import optimize
from typing import Any
import numpy as np
import open3d as o3d
from dataclasses import dataclass, field

import robotools as rt
from robotools.geometry import (
    get_6d_vector_from_affine_matrix,
    invert_homogeneous,
    get_affine_matrix_from_6d_vector,
)
from .camera import HandeyeCalibration, CamFrame


@dataclass
class DetectionDatapoint:
    img: np.ndarray
    robot_pose: np.ndarray | None
    detected: bool = False
    estimated_pose6d: np.ndarray | None = None
    corners: list[tuple[int]] = field(default_factory=lambda: [])
    inter_corners: list[tuple[float]] = field(default_factory=lambda: [])
    ids: list[list[int]] = field(default_factory=lambda: [])
    inter_ids: list[list[int]] = field(default_factory=lambda: [])


@dataclass
class CalibrationResult:
    calibration: HandeyeCalibration
    world2markers: np.ndarray


class HandeyeCalibrator:

    def __init__(
        self,
        chessboard_size: float = 0.05,
        marker_size: float = 0.04,
        n_markers=(13, 7),
        charuco_dict=aruco.DICT_ARUCO_MIP_36h12,
    ) -> None:
        self.aruco_dict = aruco.getPredefinedDictionary(charuco_dict)
        self.n_markers = n_markers
        self.marker_size = marker_size
        self.chessboard_size = chessboard_size
        self.charuco_board = aruco.CharucoBoard(
            n_markers, chessboard_size, marker_size, self.aruco_dict
        )
        self.calibration_datapoints: list[DetectionDatapoint] = []

        self.charuco_detector = cv2.aruco.CharucoDetector(self.charuco_board)  # type: ignore

    def reset(self) -> None:
        self.calibration_datapoints = []

    def capture(self, img: np.ndarray, robot_pose: np.ndarray) -> np.ndarray:
        """saves an image and robot pose for later calibration"""
        detection_result = self._detect_charuco(img, robot_pose)
        self.calibration_datapoints.append(detection_result)
        return self.draw_detection(detection_result)

    def draw_detection(
        self,
        detection_result: DetectionDatapoint,
    ) -> np.ndarray:
        img = detection_result.img.copy()  # dont change original image in calibration results!

        if detection_result.detected:
            corners = detection_result.corners
            ids = detection_result.ids

            if len(corners) > 0:
                cv2.aruco.drawDetectedCornersCharuco(img, corners, ids)  # type: ignore

        return img

    def get_line_from_poses(self, start: np.ndarray, end: np.ndarray) -> o3d.geometry.LineSet:
        # generate open3d line from 4x3 poses
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector([start[:3, 3], end[:3, 3]])
        line.lines = o3d.utility.Vector2iVector([[0, 1]])
        line.colors = o3d.utility.Vector3dVector([[1, 0, 0], [1, 0, 0]])
        return line

    def visualize_calibration(
        self,
        world2markers: np.ndarray,
        extrinsics: np.ndarray,
        intrinsics: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> None:

        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
        markers = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        markers.transform(world2markers)
        vis = [origin, markers]

        vis.append(self.get_line_from_poses(np.eye(4), world2markers))

        for cal_result in self.calibration_datapoints:
            world2robot = cal_result.robot_pose
            world2cam = world2robot @ extrinsics
            cam2markers = get_affine_matrix_from_6d_vector(
                "Rodriguez", cal_result.estimated_pose6d
            )
            world2markers_cam = world2cam @ cam2markers

            robot = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            robot.transform(world2robot)
            camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            camera.transform(world2cam)
            markers_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            markers_cam.transform(world2markers_cam)
            vis.append(robot)
            vis.append(camera)
            vis.append(markers_cam)
            vis.append(self.get_line_from_poses(np.eye(4), world2robot))
            vis.append(self.get_line_from_poses(world2robot, world2cam))
            vis.append(self.get_line_from_poses(world2cam, world2markers_cam))

            # draw charuco
            img = self.draw_detection(cal_result)

            optimized = img.copy()
            raw = img.copy()

            cv2.putText(optimized, "Optimized", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # type: ignore
            cv2.putText(raw, "Raw", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # type: ignore

            # draw optimized calibration result
            cam2monitor = invert_homogeneous(cal_result.robot_pose @ extrinsics) @ world2markers
            rvec, _ = cv2.Rodrigues(cam2monitor[:3, :3])  # type: ignore
            tvec = cam2monitor[:3, 3]
            optimized = cv2.drawFrameAxes(optimized, intrinsics, dist_coeffs, rvec, tvec, 0.1, 3)  # type: ignore

            # draw pose of this specific image
            if cal_result.estimated_pose6d is not None:
                mat = get_affine_matrix_from_6d_vector("Rodriguez", cal_result.estimated_pose6d)
                rvec, _ = cv2.Rodrigues(mat[:3, :3])  # type: ignore
                tvec = mat[:3, 3]
                raw = cv2.drawFrameAxes(raw, intrinsics, dist_coeffs, rvec, tvec, 0.1, 3)  # type: ignore

                # estimated = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                # estimated.transform(cal_result.robot_pose @ extrinsics @ mat)
                # # draw line from robot pose to estimated
                # line = o3d.geometry.LineSet()
                # line.points = o3d.utility.Vector3dVector([cal_result.robot_pose[:3, 3], tvec])
                # line.lines = o3d.utility.Vector2iVector([[0, 1]])
                # line.colors = o3d.utility.Vector3dVector([[1, 0, 0], [1, 0, 0]])
                # vis.append(line)
                # vis.append(estimated)

                print(f"optimized:\n{cam2monitor}\nvs per img:\n{mat}")

            # stack next to each other
            annotated = np.hstack([raw, optimized])

            cv2.imshow("Calibration", annotated[::2, ::2, ::-1])
            key = cv2.waitKey(0)
            if key == ord("q"):
                break
        o3d.visualization.draw_geometries(vis)

    def calibrate(self, extrinsic_guess: np.ndarray = np.eye(4)) -> CalibrationResult:

        self.calibration_datapoints = [x for x in self.calibration_datapoints if x.detected]

        print(f"Calibrating from {len(self.calibration_datapoints)} datapoints")

        assert len(self.calibration_datapoints) > 0, "No calibration data available"

        inter_corners = [x.inter_corners for x in self.calibration_datapoints]
        inter_ids = [x.inter_ids for x in self.calibration_datapoints]
        robot_poses = [x.robot_pose for x in self.calibration_datapoints]
        assert len(robot_poses) == len(self.calibration_datapoints), "Robot poses missing"

        assert len(inter_corners) > 0, "No charuco corners detected"

        image_size = (
            self.calibration_datapoints[0].img.shape[0],
            self.calibration_datapoints[0].img.shape[1],
        )

        cameraMatrixInit = np.array(
            [
                [2500, 0.0, image_size[1] / 2.0],
                [0.0, 2500.0, image_size[0] / 2.0],
                [0.0, 0.0, 1.0],
            ]
        )

        distCoefficientsInit = np.zeros((5, 1))
        flags = (
            cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO  # type: ignore
        )

        logging.info("Calibrating intrinsics...")

        obj_points = []
        img_points = []
        for _corner, _id in zip(inter_corners, inter_ids):
            obj_point, img_point = self.charuco_board.matchImagePoints(_corner, _id)
            obj_points.append(obj_point)
            img_points.append(img_point)

        ret, camera_matrix, dist_coefficients, rvecs, tvecs = cv2.calibrateCamera(
            objectPoints=obj_points,
            imagePoints=img_points,
            imageSize=image_size,
            cameraMatrix=cameraMatrixInit,
            distCoeffs=distCoefficientsInit,
            rvecs=None,
            tvecs=None,
            flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9),  # type: ignore
        )

        assert ret, "Calibration failed"

        for cal_result, rvec, tvec in zip(self.calibration_datapoints, rvecs, tvecs):
            cal_result.estimated_pose6d = np.concatenate([tvec, rvec], axis=0)[:, 0].astype(
                np.float64
            )

        logging.info("Done")

        logging.info("Calibrating extrinsics...")
        camera_poses = [
            np.concatenate([tvec, rvec], axis=0)[:, 0] for tvec, rvec in zip(tvecs, rvecs)
        ]
        ret = self._optimize_handeye_matrix(
            camera_poses, robot_poses, initial_guess=extrinsic_guess
        )
        logging.info("Done")
        logging.info(f"Optimality: {ret['optimality']}")
        logging.info(f"Cost:       {ret['cost']}")

        x = ret["x"]
        extrinsic_matrix = invert_homogeneous(get_affine_matrix_from_6d_vector("xyz", x[:6]))
        print("Extrinsic matrix:\n", extrinsic_matrix)
        world2markers = invert_homogeneous(get_affine_matrix_from_6d_vector("xyz", x[6:]))

        # try OpenCV version
        # R_gripper2base = [x[:3, :3] for x in robot_poses]
        # t_gripper2base = [x[:3, 3] for x in robot_poses]
        # R_target2cam = rvecs
        # t_target2cam = tvecs
        # R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(  # type: ignore
        #     R_gripper2base,
        #     t_gripper2base,
        #     R_target2cam,
        #     t_target2cam,
        #     method=cv2.CALIB_HAND_EYE_TSAI,  # type: ignore
        # )
        # extrinsic_matrix = np.eye(4)
        # extrinsic_matrix[:3, :3] = R_cam2gripper
        # extrinsic_matrix[:3, 3] = np.reshape(t_cam2gripper, (3,))
        # print("Extrinsic matrix:\n", extrinsic_matrix)

        return CalibrationResult(
            calibration=rt.camera.HandeyeCalibration(
                intrinsic_matrix=camera_matrix,
                dist_coeffs=dist_coefficients,
                extrinsic_matrix=extrinsic_matrix,
            ),
            world2markers=world2markers,
        )

    def _detect_charuco(self, img: np.ndarray, robot_pose: np.ndarray) -> DetectionDatapoint:

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # type: ignore

        corners, ids, markers_corners, marker_ids = self.charuco_detector.detectBoard(  # type: ignore
            gray
        )
        if corners is None:
            return DetectionDatapoint(img=img.copy(), robot_pose=robot_pose.copy())

        cal_result = DetectionDatapoint(
            img.copy(),
            robot_pose.copy(),
            detected=True,
            corners=corners,
            ids=ids,
            inter_corners=corners,
            inter_ids=ids,
        )
        return cal_result

    def _optimize_handeye_matrix(
        self,
        camera_poses: list[np.ndarray],
        robot_poses: list[np.ndarray],
        initial_guess: np.ndarray,
    ) -> Any:
        # camera2tool_t = np.zeros((6,))
        # camera2tool_t[5] = np.pi  # initialize with 180° around z
        # get_affine_matrix_from_6d_vector("xyz", x[:6])
        camera2tool_t = get_6d_vector_from_affine_matrix("xyz", initial_guess)

        marker2wc_t = np.zeros((6,))
        marker2camera_t = [
            invert_homogeneous(get_affine_matrix_from_6d_vector("Rodriguez", x))
            for x in camera_poses
        ]

        tool2wc_t = [invert_homogeneous(x) for x in robot_poses]  # already homogeneous matrix

        x0 = np.array([camera2tool_t, marker2wc_t]).reshape(12)

        def residual(
            x: np.ndarray,
            tool2wc: np.ndarray,
            marker2camera: np.ndarray,
        ) -> np.ndarray:
            camera2tool = get_affine_matrix_from_6d_vector("xyz", x[:6])
            marker2wc = get_affine_matrix_from_6d_vector("xyz", x[6:])
            return res_func(marker2camera, tool2wc, camera2tool, marker2wc)

        def res_func(
            marker2camera: np.ndarray,
            tool2wc: np.ndarray,
            camera2tool: np.ndarray,
            marker2wc: np.ndarray,
        ) -> np.ndarray:
            res = []
            for i in range(len(marker2camera)):
                res += single_res_func(marker2camera[i], tool2wc[i], camera2tool, marker2wc)
            return np.array(res).reshape(16 * len(marker2camera))

        def single_res_func(
            marker2camera: np.ndarray,
            tool2wc: np.ndarray,
            camera2tool: np.ndarray,
            marker2wc: np.ndarray,
        ) -> list[np.ndarray]:
            res_array = marker2camera @ camera2tool @ tool2wc - marker2wc
            return [res_array.reshape((16,))]

        ret = optimize.least_squares(
            residual,
            x0,
            kwargs={"marker2camera": marker2camera_t, "tool2wc": tool2wc_t},
        )
        return ret
