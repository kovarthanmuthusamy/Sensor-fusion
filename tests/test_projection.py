from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from sensor_fusion.data.projection import (
    build_lidar_to_camera,
    make_sparse_depth,
    project_to_image,
)

IDENTITY = {"translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]}
HEIGHT, WIDTH = 900, 1600
IMAGE_SIZE = (HEIGHT, WIDTH)
# Focal length 1000 px, principal point (800, 450): easy to compute by hand.
INTRINSIC = np.array([[1000.0, 0.0, 800.0], [0.0, 1000.0, 450.0], [0.0, 0.0, 1.0]])
SQRT_HALF = float(np.sqrt(0.5))
DATASET = Path(__file__).parents[1] / "dataset" / "v1.0-mini"


def _record(translation: list[float], rotation: list[float]) -> dict[str, list[float]]:
    return {"translation": translation, "rotation": rotation}


def _apply(transform: np.ndarray, point: list[float]) -> np.ndarray:
    return (transform @ np.array([*point, 1.0]))[:3]


# --- build_lidar_to_camera -------------------------------------------------------------


def test_chain_of_identities_is_identity() -> None:
    chain = build_lidar_to_camera(IDENTITY, IDENTITY, IDENTITY, IDENTITY)
    np.testing.assert_allclose(chain, np.eye(4), atol=1e-12)


def test_quaternion_is_read_as_w_x_y_z() -> None:
    # [cos45, 0, 0, sin45] is +90 degrees about z in [w, x, y, z] order: x-axis -> y-axis.
    # Read wrongly as [x, y, z, w] it would be 90 degrees about x, which leaves the x-axis alone.
    quarter_turn = _record([0.0, 0.0, 0.0], [SQRT_HALF, 0.0, 0.0, SQRT_HALF])
    chain = build_lidar_to_camera(quarter_turn, IDENTITY, IDENTITY, IDENTITY)
    np.testing.assert_allclose(_apply(chain, [1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)


def test_rotation_matches_pyquaternion() -> None:
    pyquaternion = pytest.importorskip("pyquaternion")
    rng = np.random.default_rng(0)
    for _ in range(20):
        quaternion = rng.normal(size=4)
        expected = pyquaternion.Quaternion(quaternion).rotation_matrix
        chain = build_lidar_to_camera(
            _record([0.0, 0.0, 0.0], quaternion.tolist()), *[IDENTITY] * 3
        )
        np.testing.assert_allclose(chain[:3, :3], expected, atol=1e-12)


def test_car_motion_between_firings_shifts_the_point() -> None:
    # The car is at the origin when the LiDAR fires and 1 m further along x when the camera
    # fires. A point 10 m ahead at LiDAR time is therefore only 9 m ahead at camera time.
    moved_car = _record([1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    chain = build_lidar_to_camera(IDENTITY, IDENTITY, moved_car, IDENTITY)
    np.testing.assert_allclose(_apply(chain, [10.0, 0.0, 0.0]), [9.0, 0.0, 0.0], atol=1e-12)


def test_camera_calibration_is_applied_inverted() -> None:
    # A camera mounted turned +90 degrees about z sees the car's +x axis along its own -y axis.
    turned_camera = _record([0.0, 0.0, 0.0], [SQRT_HALF, 0.0, 0.0, SQRT_HALF])
    chain = build_lidar_to_camera(IDENTITY, IDENTITY, IDENTITY, turned_camera)
    np.testing.assert_allclose(_apply(chain, [1.0, 0.0, 0.0]), [0.0, -1.0, 0.0], atol=1e-12)


# --- project_to_image ------------------------------------------------------------------


def test_point_10m_ahead_hits_the_principal_point() -> None:
    pixels, depths = project_to_image(
        np.array([[0.0, 0.0, 10.0]]), np.eye(4), INTRINSIC, IMAGE_SIZE
    )
    np.testing.assert_allclose(pixels, [[800.0, 450.0]])
    np.testing.assert_allclose(depths, [10.0])


def test_off_axis_point_follows_the_pinhole_formula() -> None:
    # u = 1000 * 1 / 10 + 800 = 900;  v = 1000 * 0.5 / 10 + 450 = 500
    pixels, _ = project_to_image(np.array([[1.0, 0.5, 10.0]]), np.eye(4), INTRINSIC, IMAGE_SIZE)
    np.testing.assert_allclose(pixels, [[900.0, 500.0]])


def test_depth_is_camera_z_not_straight_line_range() -> None:
    _, depths = project_to_image(np.array([[3.0, 0.0, 4.0]]), np.eye(4), INTRINSIC, IMAGE_SIZE)
    np.testing.assert_allclose(depths, [4.0])  # the range would be 5.0


def test_points_behind_or_too_close_are_dropped() -> None:
    points = np.array([[0.0, 0.0, -5.0], [0.0, 0.0, 0.5], [0.0, 0.0, 10.0]])
    pixels, depths = project_to_image(points, np.eye(4), INTRINSIC, IMAGE_SIZE, min_depth=1.0)
    assert depths.tolist() == [10.0]
    assert pixels.shape == (1, 2)


def test_points_outside_the_image_are_dropped() -> None:
    # u = 1000 * 100 / 10 + 800 is far to the right of the 1600-pixel-wide image.
    points = np.array([[100.0, 0.0, 10.0], [0.0, 0.0, 10.0]])
    pixels, _ = project_to_image(points, np.eye(4), INTRINSIC, IMAGE_SIZE)
    np.testing.assert_allclose(pixels, [[800.0, 450.0]])


def test_pixel_that_rounds_up_to_the_image_edge_is_dropped() -> None:
    # u = 1599.999999999 is inside the image in float64 but becomes 1600.0 in float32,
    # which would be column 1600 of a 1600-column image.
    almost_edge = INTRINSIC.copy()
    almost_edge[0, 2] = 1599.999999999
    pixels, _ = project_to_image(np.array([[0.0, 0.0, 10.0]]), np.eye(4), almost_edge, IMAGE_SIZE)
    assert pixels.shape == (0, 2)


def test_points_with_four_columns_are_rejected() -> None:
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        project_to_image(np.zeros((5, 4)), np.eye(4), INTRINSIC, IMAGE_SIZE)


def test_transposed_points_are_rejected() -> None:
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        project_to_image(np.zeros((3, 100)), np.eye(4), INTRINSIC, IMAGE_SIZE)


def test_non_positive_min_depth_is_rejected() -> None:
    with pytest.raises(ValueError, match="min_depth"):
        project_to_image(np.zeros((1, 3)), np.eye(4), INTRINSIC, IMAGE_SIZE, min_depth=0.0)


# --- make_sparse_depth -----------------------------------------------------------------


def test_depth_lands_in_the_pixel_that_contains_the_point() -> None:
    image = make_sparse_depth(np.array([[10.7, 20.2]]), np.array([7.5]), IMAGE_SIZE)
    assert image.shape == IMAGE_SIZE
    assert image.dtype == np.float32
    assert image[20, 10] == 7.5  # row = v, column = u
    assert np.count_nonzero(image) == 1


def test_nearest_point_wins_whatever_the_input_order() -> None:
    pixels = np.array([[5.1, 5.1], [5.9, 5.9], [5.5, 5.5]])  # all in pixel (row 5, column 5)
    depths = np.array([30.0, 12.0, 20.0])
    for order in ([0, 1, 2], [2, 1, 0], [1, 0, 2]):
        image = make_sparse_depth(pixels[order], depths[order], IMAGE_SIZE)
        assert image[5, 5] == 12.0
        assert np.count_nonzero(image) == 1


def test_no_points_gives_an_all_zero_image() -> None:
    image = make_sparse_depth(np.zeros((0, 2)), np.zeros(0), IMAGE_SIZE)
    assert image.shape == IMAGE_SIZE
    assert not image.any()


def test_pixels_outside_the_image_are_rejected() -> None:
    with pytest.raises(ValueError, match="outside the image"):
        make_sparse_depth(np.array([[1600.0, 10.0]]), np.array([5.0]), IMAGE_SIZE)


def test_zero_depth_is_rejected_because_zero_means_no_return() -> None:
    with pytest.raises(ValueError, match="depths must be > 0"):
        make_sparse_depth(np.array([[1.0, 1.0]]), np.array([0.0]), IMAGE_SIZE)


# --- shape guards at the function boundaries -------------------------------------------


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: build_lidar_to_camera(_record([0.0] * 3, [1.0, 0.0, 0.0]), *[IDENTITY] * 3),
            "4 numbers",
        ),
        (
            lambda: project_to_image(np.zeros((1, 3)), np.eye(3), INTRINSIC, IMAGE_SIZE),
            "T_cam_from_lidar",
        ),
        (
            lambda: project_to_image(np.zeros((1, 3)), np.eye(4), np.eye(4), IMAGE_SIZE),
            "intrinsic",
        ),
        (lambda: make_sparse_depth(np.zeros((2, 3)), np.zeros(2), IMAGE_SIZE), r"\(M, 2\)"),
        (lambda: make_sparse_depth(np.zeros((2, 2)), np.ones(3), IMAGE_SIZE), "depths must have"),
    ],
)
def test_wrong_shapes_are_rejected_with_a_clear_message(
    call: Callable[[], object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        call()


# --- oracle: the devkit's own projection, on real data -----------------------------------


@pytest.mark.skipif(not DATASET.exists(), reason="nuScenes-mini not found at dataset/v1.0-mini")
def test_matches_the_devkit_on_the_first_frame_of_every_mini_scene() -> None:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud

    nusc = NuScenes("v1.0-mini", dataroot=str(DATASET), verbose=False)
    for scene in nusc.scene:
        sample = nusc.get("sample", scene["first_sample_token"])
        lidar = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        cam = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        cam_calib = nusc.get("calibrated_sensor", cam["calibrated_sensor_token"])

        points = LidarPointCloud.from_file(str(DATASET / lidar["filename"])).points[:3].T
        T_cam_from_lidar = build_lidar_to_camera(
            nusc.get("calibrated_sensor", lidar["calibrated_sensor_token"]),
            nusc.get("ego_pose", lidar["ego_pose_token"]),
            nusc.get("ego_pose", cam["ego_pose_token"]),
            cam_calib,
        )
        pixels, depths = project_to_image(
            points, T_cam_from_lidar, np.array(cam_calib["camera_intrinsic"]), IMAGE_SIZE
        )
        # The devkit also drops a 1-pixel border; apply the same margin before comparing.
        margin = (
            (pixels[:, 0] > 1)
            & (pixels[:, 0] < WIDTH - 1)
            & (pixels[:, 1] > 1)
            & (pixels[:, 1] < HEIGHT - 1)
        )

        devkit_pixels, devkit_depths, _ = nusc.explorer.map_pointcloud_to_image(
            lidar["token"], cam["token"]
        )
        assert margin.sum() == devkit_pixels.shape[1], scene["name"]
        np.testing.assert_allclose(pixels[margin], devkit_pixels[:2].T, atol=0.05)
        np.testing.assert_allclose(depths[margin], devkit_depths, atol=0.02)
