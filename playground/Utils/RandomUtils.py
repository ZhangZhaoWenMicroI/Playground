"""
Random utilities for sampling in robotics and simulation.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Optional, Tuple, Any


def random_quat() -> np.ndarray:
    """
    Generate a random unit quaternion.
    Returns:
        np.ndarray: Quaternion (4,)
    """
    random_values = np.random.normal(0, 1, 4)
    norm = np.linalg.norm(random_values)
    quaternion = random_values / norm
    return quaternion


def random_axis_angle(angle_limit=None, random_state=None):
    """
    Samples an axis-angle rotation by first sampling a random axis
    and then sampling an angle. If @angle_limit is provided, the size
    of the rotation angle is constrained.

    If @random_state is provided (instance of np.random.RandomState), it
    will be used to generate random numbers.

    Args:
        angle_limit (None or float): If set, determines magnitude limit of angles to generate
        random_state (None or RandomState): RNG to use if specified

    Raises:
        AssertionError: [Invalid RNG]
    """
    if angle_limit is None:
        angle_limit = 2.0 * np.pi

    if random_state is not None:
        assert isinstance(random_state, np.random.RandomState)
        npr = random_state
    else:
        npr = np.random
    random_axis = npr.randn(3)
    random_axis /= np.linalg.norm(random_axis)
    random_angle = npr.uniform(low=0.0, high=angle_limit)
    return random_axis, random_angle


def random_points_in_sphere(radius=1.0, center=None, num_points=1, random_state=None):
    """
    Generates uniformly random points within a 3D sphere.

    Args:
        radius (float): Radius of the sphere. Must be non-negative.
        center (np.ndarray, optional): A 3D array (x, y, z) representing the sphere's center.
                                      Defaults to [0, 0, 0].
        num_points (int): The number of points to generate. Must be a positive integer.
        random_state (None or np.random.RandomState): RNG to use if specified.

    Returns:
        np.ndarray: An array of shape (num_points, 3) containing the generated points.

    Raises:
        ValueError: If radius is negative or num_points is not positive.
    """
    if radius < 0:
        raise ValueError("Radius must be non-negative.")
    if num_points <= 0:
        raise ValueError("Number of points must be positive.")

    if center is None:
        center = np.zeros(3)
    else:
        center = np.asarray(center, dtype=float)
        if center.shape != (3,):
            raise ValueError("Center must be a 3D array-like object.")

    if random_state is not None:
        assert isinstance(random_state, np.random.RandomState), "[Invalid RNG]"
        npr = random_state
    else:
        npr = np.random

    vectors = npr.normal(0, 1, (num_points, 3))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)

    # Handle cases where norm might be zero for a very small fraction of samples
    # (though highly unlikely with float precision for normal distribution)
    norms[norms == 0] = 1e-12  # Replace 0 with a tiny number to avoid division by zero

    unit_vectors = vectors / norms

    # Scale to fill the volume uniformly
    # For uniform points *within* the sphere, we need to scale by r * (u^(1/dim))
    # where u is a uniform random variable in [0, 1] and dim is the dimension (3 for 3D)
    radii_scaling = radius * (npr.uniform(0, 1, size=(num_points, 1)) ** (1 / 3.0))

    points = unit_vectors * radii_scaling

    # Translate points to the desired center
    points += center

    return points


def random_points_in_box(dims, pose=None, num_points=1, random_state=None):
    """
    Generates uniformly random points within a 3D rectangular box with a specified pose.

    Args:
        dims (np.ndarray or list): A 3-element array/list [length_x, width_y, height_z]
                                   representing the dimensions of the box along its local axes.
                                   All dimensions must be non-negative.
        pose (dict, optional): A dictionary specifying the box's pose.
                               Keys:
                                 - 'xyz' (np.ndarray or list): A 3D array (x, y, z) for translation.
                                   Defaults to [0, 0, 0].
                                 - 'rpy' (np.ndarray or list): A 3D array (roll, pitch, yaw) in radians.
                                   Defaults to [0, 0, 0].
                                 - 'quat' (np.ndarray or list): A 4D array (w, x, y, z) for rotation.
                                   If 'rpy' is provided, 'quat' is ignored.
                                   Defaults to [1, 0, 0, 0] (no rotation).
        num_points (int): The number of points to generate. Must be a positive integer.
        random_state (None or np.random.RandomState): RNG to use if specified.

    Returns:
        np.ndarray: An array of shape (num_points, 3) containing the generated points.

    Raises:
        ValueError: If any dimension is negative, or num_points is not positive,
                    or pose parameters have incorrect shapes.
    """
    dims = np.asarray(dims, dtype=float)
    if dims.shape != (3,) or np.any(dims < 0):
        raise ValueError(
            "Dimensions must be a 3-element array/list with non-negative values."
        )
    if num_points <= 0:
        raise ValueError("Number of points must be positive.")

    if random_state is not None:
        assert isinstance(random_state, np.random.RandomState), "[Invalid RNG]"
        npr = random_state
    else:
        npr = np.random

    # Default pose
    translation = np.zeros(3)
    rotation = R.from_quat([0, 0, 0, 1])  # (x,y,z,w) order for scipy

    if pose is not None:
        if "xyz" in pose:
            translation = np.asarray(pose["xyz"], dtype=float)
            if translation.shape != (3,):
                raise ValueError("Pose 'xyz' must be a 3D array-like object.")

        if "rpy" in pose:
            rpy = np.asarray(pose["rpy"], dtype=float)
            if rpy.shape != (3,):
                raise ValueError("Pose 'rpy' must be a 3D array-like object.")
            rotation = R.from_euler(
                "xyz", rpy, degrees=False
            )  # Assuming XYZ Euler sequence for RPY

        elif "quat" in pose:
            quat_w_xyz = np.asarray(pose["quat"], dtype=float)
            if quat_w_xyz.shape != (4,):
                raise ValueError(
                    "Pose 'quat' must be a 4D array-like object (w,x,y,z)."
                )
            # scipy.spatial.transform.Rotation.from_quat expects (x, y, z, w)
            rotation = R.from_quat(
                [quat_w_xyz[1], quat_w_xyz[2], quat_w_xyz[3], quat_w_xyz[0]]
            )

    # Generate points in a canonical axis-aligned box centered at origin
    half_dims = dims / 2.0
    min_coords_local = -half_dims
    max_coords_local = half_dims

    local_points = npr.uniform(
        low=min_coords_local, high=max_coords_local, size=(num_points, 3)
    )

    # Apply rotation
    rotated_points = rotation.apply(local_points)

    # Apply translation
    world_points = rotated_points + translation

    return world_points


def random_rotation_matrix(random_state=None):
    """
    Generates a uniformly random 3x3 rotation matrix (Haar measure).
    Uses SVD of a random matrix to ensure uniform distribution over SO(3).

    Args:
        random_state (None or np.random.RandomState): RNG to use if specified.

    Returns:
        np.ndarray: A 3x3 uniformly random rotation matrix.
    """
    if random_state is not None:
        assert isinstance(random_state, np.random.RandomState), "[Invalid RNG]"
        npr = random_state
    else:
        npr = np.random

    A = npr.normal(0, 1, (3, 3))
    Q, R_decomp = np.linalg.qr(A)

    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q
