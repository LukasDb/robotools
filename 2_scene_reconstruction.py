import cv2
import tensorflow as tf

from robotools.geometry import invert_homogeneous

for dev in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(dev, True)
import numpy as np
import yaml
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import click
import asyncio
import matplotlib.pyplot as plt
import open3d as o3d


import simpose as sp
import robotools as rt
from robotools.trajectory import SphericalTrajectory, TrajectoryExecutor


async def async_main(capture: bool, output: Path) -> None:
    scene = rt.Scene()
    scene.from_config(yaml.safe_load(open("scene.yaml")))

    # extract names from scene.yaml
    # cam: rt.camera.Camera = scene._entities["Realsense_121622061798"]
    cam: rt.camera.Camera = scene._entities["ZED-M-12049762"]

    bg: rt.utility.BackgroundMonitor = scene._entities["Background Monitor"]
    robot: rt.robot.Robot = scene._entities["crx"]

    print(f"Loaded scene from file: {cam.calibration}")
    trajectory = SphericalTrajectory(
        thetas=np.linspace(-110, 270, 8, endpoint=False).tolist(),
        pitchs=[40, 50, 65, 75],
        radius=[0.4],
        center_point=(0.68, 0.16, -0.16),
        view_jitter=(5, 5, 5),
    )

    if capture:
        executor = TrajectoryExecutor()
        trajectory.visualize()

        input("Please move the window to the second screen and press enter")
        bg.setup_window()
        bg.set_to_depth_texture()

        # 2) acquire
        output.mkdir(parents=True, exist_ok=True)
        writer_params = sp.writers.WriterConfig(
            output_dir=output,
            overwrite=True,
            start_index=0,
            end_index=len(trajectory) - 1,
        )
        with sp.writers.TFRecordWriter(writer_params, None) as writer:
            print("Starting capturing...")
            async for step in executor.execute(robot, trajectory, cam=cam):
                # cam_rot = await cam.get_orientation()
                # cam_rot = cam_rot.as_quat()

                # cam_pos = await cam.get_position()

                cam_pose = await robot.get_pose() @ cam.calibration.extrinsic_matrix
                cam_rot = R.from_matrix(cam_pose[:3, :3]).as_quat()
                cam_pos = cam_pose[:3, 3]

                frame = cam.get_frame()

                rp = sp.RenderProduct(
                    rgb=frame.rgb,
                    rgb_R=frame.rgb_R,
                    depth=frame.depth,
                    depth_R=frame.depth_R,
                    intrinsics=cam.calibration.intrinsic_matrix,
                    cam_position=cam_pos,
                    cam_quat_xyzw=cam_rot,
                )
                writer.write_data(step, render_product=rp)
    bg.disable()
    # 3) reconstruct
    # output = Path("~/data/real/occluded_cpsduck_realsense_121622061798").expanduser()
    # output = Path("~/data/real/single_cpsduck_realsense_121622061798").expanduser()

    tfds = sp.data.TFRecordDataset
    dataset = tfds.get(
        output,
        get_keys=[
            tfds.CAM_LOCATION,
            tfds.CAM_ROTATION,
            tfds.CAM_MATRIX,
            tfds.OBJ_POS,
            tfds.OBJ_ROT,
            tfds.RGB,
            tfds.DEPTH,
        ],
    ).enumerate()
    # only select every nth image (for demo data)
    # the dataset contains always secquences of 5 images with only backdrop changing
    # nth = 5
    # dataset = dataset.filter(lambda i, _: tf.math.floormod(i, nth) == 0)

    scene_integration = rt.SceneIntegration(use_color=True)

    if True:
        for i, data in tqdm(dataset, total=len(trajectory), desc="Reconstructing..."):
            t = data[tfds.CAM_LOCATION]
            r = R.from_quat(data[tfds.CAM_ROTATION].numpy())  # xyzw quaternion
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = r.as_matrix()
            extrinsic[:3, 3] = t.numpy()

            scene_integration.integrate(
                data[tfds.RGB].numpy(),
                data[tfds.DEPTH].numpy(),
                data[tfds.CAM_MATRIX].numpy(),
                extrinsic,
            )
        scene_integration.save()
    else:
        scene_integration.load()

    print("Done")
    mesh = scene_integration.vbg.extract_triangle_mesh()
    print("Mesh extracted. Now processing...")
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    mesh = mesh.to_legacy()

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    o3d.visualization.draw_geometries([mesh,origin])

    n_prev = 5
    fig, axs = plt.subplots(n_prev, 3)
    mean_diffs = []
    outlier_ratios = []
    for step, data in tqdm(dataset, total=len(trajectory), desc="Analyzing..."):
        extrinsics = np.eye(4)
        extrinsics[:3, 3] = data[tfds.CAM_LOCATION].numpy()
        extrinsics[:3, :3] = R.from_quat(data[tfds.CAM_ROTATION].numpy()).as_matrix()
        extrinsics = invert_homogeneous(extrinsics)
        h, w = data[tfds.DEPTH].shape[:2]

        depth_reconst = scene_integration.render_depth(
            w, h, data[tfds.CAM_MATRIX].numpy().astype(np.float64), extrinsics.astype(np.float64)
        )
        depth_raw = data[tfds.DEPTH].numpy() * 1000

        # simple hole filler
        depth_fused = np.copy(depth_reconst)
        depth_fused[depth_reconst < 100] = depth_raw[depth_reconst < 10]  # is in mm!

        diff = np.abs(depth_reconst - depth_raw)
        threshold = 10
        mean_diff = np.mean(diff[diff < threshold])
        mean_diffs.append(mean_diff)
        num_outliers = np.count_nonzero(diff > threshold)
        outlier_ratios.append(num_outliers / diff.size)

        if step >= n_prev:
            continue

        colorized_depth = o3d.t.geometry.Image(
            depth_reconst
        )  # .colorize_depth(1000.0, d_min, d_max)
        axs[step, 0].imshow(colorized_depth.as_tensor().cpu().numpy())
        axs[step, 0].set_title("depth (Reconstructed)")

        colorized_depth_raw = o3d.t.geometry.Image(
            depth_raw
        )  # .colorize_depth(1000.0, d_min, d_max)
        axs[step, 1].imshow(colorized_depth_raw.as_tensor().cpu().numpy())
        axs[step, 1].set_title("depth (Realsense)")

        colorized_depth_fused = o3d.t.geometry.Image(
            depth_fused
        )  # .colorize_depth(            1000.0, d_min, d_max        )
        axs[step, 2].imshow(colorized_depth_fused.as_tensor().cpu().numpy())
        axs[step, 2].set_title("depth (fused)")

        continue

        print(f"Mean diff: {mean_diff}")
        colorized_diff = o3d.t.geometry.Image(diff).colorize_depth(1000.0, 0, 0.1)
        axs[2].imshow(colorized_diff.as_tensor().cpu().numpy())
        axs[2].set_title("Difference")
        plt.show(block=False)
        input()
        axs[0].clear()
        axs[1].clear()
        axs[2].clear()
    print(f"Mean mean diff: {np.mean(mean_diffs)}")
    print(f"Mean num outliers: {np.mean(outlier_ratios)*100:.2f}%")
    plt.show()


@click.command()
@click.option("--capture", is_flag=True)
@click.option("--output", type=click.Path(path_type=Path), default="data/scene_construction")
def main(capture: bool, output: Path):
    asyncio.run(async_main(capture, output))


if __name__ == "__main__":
    main()
