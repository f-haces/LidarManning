import random, cv2, laspy

import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from skspatial.objects import Plane

import torch
from torch.nn.utils.rnn import pad_sequence

import rasterio
from rasterio.transform import from_origin

def detrend_point_cloud(points, plane_sample=10000):
    """
    Remove the trend component from a 3D point cloud by subtracting the best-fit plane.

    Parameters:
    - points (numpy.ndarray): Array of shape (N, 3) representing the 3D coordinates of points in the point cloud.
    - plane_sample (int, optional): Number of points to sample for estimating the best-fit plane. Default is 10000.

    Returns:
    - numpy.ndarray: Point cloud with the trend component removed.

    This function estimates the best-fit plane to the point cloud using a subset of points sampled randomly.
    It then subtracts the plane from each point in the cloud to detrend it.
    """

    plane = Plane.best_fit(points[random.sample(range(1, points.shape[0]), plane_sample), :])
    a, b, c, d = plane.cartesian()
    z_plane = (a * points[:, 0] + b * points[:, 1] + d) / (-1 * c)
    points[:, 2] = points[:, 2] - z_plane
    return points

def normalizePoints(points):
    """
    Normalize a set of 2D or 3D points by detrending and rotating them.

    Parameters:
    - points (numpy.ndarray): Array of shape (N, 2) or (N, 3) representing the 2D or 3D coordinates of points.

    Returns:
    - numpy.ndarray: Normalized points.

    This function first finds the minimum bounding rectangle of the input points, 
    then rotates the points to align with the x-axis of the coordinate system.
    Finally, it detrends the rotated points by subtracting the best-fit plane.

    Note: This function relies on the `detrend_point_cloud` function to detrend the point cloud.
    """
    # DETRENDS POINT CLOUD BY PLANE
    rect = cv2.minAreaRect(points[:, :2].astype(np.float32))

    angle = rect[2]
    center = rect[0]
    size   = rect[1]
    
    if size[0] > size[1]:
        angle = angle + 90
    
    ar = -1 * np.radians(angle)
    # ROTATE ABOUT Z AXIS
    rotate    = np.array([
            [np.cos(ar), -1 * np.sin(ar), 0],
            [np.sin(ar),      np.cos(ar), 0],
            [         0,               0, 1]])
    
    out = points.copy()
    
    out[:, 0] = out[:, 0] - center[0]
    out[:, 1] = out[:, 1] - center[1]

    # test = translate @ test.T
    out = rotate @ out.T
    
    out = out.T
    
    return detrend_point_cloud(out)
    
def mixPointClouds(roughnesses, points, proportions=None, min_sample_size=5, max_sample_size=100):
    """
    Mix multiple point clouds based on their roughness values and proportions.

    Parameters:
    - roughnesses (numpy.ndarray): Array of shape (N, 1) representing the roughness values corresponding to point clouds.
    - points (list): List of numpy arrays, each representing a point cloud.
    - proportions (array-like, optional): Proportions of each point cloud in the output mix. If None, random proportions are generated. Default is None.
    - min_sample_size (int, optional): Minimum number of points in the output point cloud. Default is 5.
    - max_sample_size (int, optional): Maximum number of points in the output point cloud. Default is 100.

    Returns:
    - numpy.ndarray: Mixed point cloud.

    This function creates a mixed point cloud by sampling points from each input point cloud based on their proportions
    and then stacking them together. The number of points in the output point cloud is randomly determined between
    `min_sample_size` and `max_sample_size`. The roughness of the output point cloud is computed as a weighted sum of
    the input point clouds' roughness values.

    Note: The `roughnesses` array should have the same length as the `points` list.
    """
    # ROUGHNESSES IS A NUMPY ARRAY OF N x 1, WHERE N IS THE NUMBER OF ROUGHNESSES CORRESPONDING TO POINT CLOUDS
    # POINTS IS A LIST OF ALL THE POINT CLOUDS
    
    # HOW MANY TOTAL POINTS IN OUTPUT PC
    n = np.random.randint(min_sample_size, max_sample_size)
    
    if proportions is None: 
        if roughnesses.shape[0] == 1:
            proportions = 1
        else:
            proportions = np.random.random(roughnesses.shape[0])
        
    # NORMALIZE TO SUM PROPORTIONS TO ONE, THEN CALCULATE ACTUAL N FOR EACH PC
    props_perc = np.array(proportions) / np.sum(proportions)
    props_n    = np.int32(n * props_perc)
    
    # CALCULATE OUTPUT ROUGHNESS
    rough = np.sum(props_perc * roughnesses)
    
    out = []
    
    # FOR EACH POINT CLOUD
    for i, p in enumerate(points):
        # CHOICE IS SOOOOO SLOW. USE RANDINT INSTEAD
        # choices = np.random.choice(p.shape[0], size=props_n[i], replace=False)
        choices = np.random.randint(0, p.shape[0], size=(props_n[i]))
        out.append(p[choices, :])
        
    out = np.vstack(out)
    
    return out, rough
    
def random_rotate(point_cloud):
    """
    Apply a random rotation to a 3D point cloud.

    Parameters:
    - point_cloud (numpy.ndarray): Array of shape (N, 3) representing the 3D coordinates of points in the point cloud.

    Returns:
    - numpy.ndarray: Point cloud after applying the random rotation.

    This function generates a random rotation angle and creates a rotation matrix for rotating points about the z-axis.
    It then applies the rotation to the input point cloud and returns the rotated point cloud.

    Note: This function depends on the `meanShift` function for further processing of the rotated point cloud.
    """
    # Generate a random rotation angle
    random_angle = np.random.uniform(0, 2 * np.pi)

    # Define the rotation matrix for a rotation about the z-axis
    rotation_matrix = np.array([[np.cos(random_angle), -np.sin(random_angle), 0],
                                [np.sin(random_angle), np.cos(random_angle), 0],
                                [0, 0, 1]])

    # Apply the rotation to the point cloud
    rotated_point_cloud = np.dot(point_cloud, rotation_matrix.T)

    return meanShift(rotated_point_cloud)
    
def calculate_trendline_angle(x, y):
    """
    Calculate the angle of the trendline from given x and y coordinates.

    Parameters:
    - x (numpy.ndarray): Array of x coordinates.
    - y (numpy.ndarray): Array of y coordinates.

    Returns:
    - float: Angle of the trendline in degrees.
    """

    # Fit a linear regression model
    coefficients = np.polyfit(x, y, 1)

    # Extract the slope (m) from the coefficients
    slope = coefficients[0]

    # Calculate the angle in radians
    angle_rad = np.arctan(slope)

    # Convert the angle to degrees
    angle_deg = np.degrees(angle_rad)

    return angle_deg
    
def meanShift(point_cloud):
    """
    Apply mean shift to normalize a 3D point cloud.

    Parameters:
    - point_cloud (numpy.ndarray): Array of shape (N, 3) representing the 3D coordinates of points.

    Returns:
    - numpy.ndarray: Normalized point cloud.
    """

    # Calculate the mean in each dimension
    mean_values = np.mean(point_cloud, axis=0)

    # Subtract the mean from each dimension to normalize
    normalized_point_cloud = point_cloud - mean_values

    return normalized_point_cloud
    
def group_points_by_grid(point_cloud, side_length):
    """
    Group points from a 2D point cloud based on a grid defined by a given side length.

    Parameters:
    - point_cloud (numpy.ndarray): Array of shape (N, 2) representing the 2D coordinates of points.
    - side_length (float): Length of the side of each grid cell.

    Returns:
    - list: List of NumPy arrays containing groups of points.

    This function divides the 2D point cloud into grid cells based on the specified side length,
    and assigns each point to the corresponding cell. It returns a list of arrays, each containing
    the points belonging to a particular grid cell.
    """


    # Extract x and y coordinates from the point cloud
    x_coords = point_cloud[:, 0]
    y_coords = point_cloud[:, 1]

    # Calculate the number of divisions in x and y directions
    num_x_divisions = int(np.ceil((x_coords.max() - x_coords.min()) / side_length))
    num_y_divisions = int(np.ceil((y_coords.max() - y_coords.min()) / side_length))

    # Initialize a list to store groups of points
    point_groups = [[] for _ in range(num_x_divisions * num_y_divisions)]

    # Assign each point to the corresponding group
    for i in tqdm(range(point_cloud.shape[0])):
        x_index = int((x_coords[i] - x_coords.min()) // side_length)
        y_index = int((y_coords[i] - y_coords.min()) // side_length)
        group_index = y_index * num_x_divisions + x_index
        point_groups[group_index].append(point_cloud[i])

    # Convert the lists to NumPy arrays
    point_groups = [np.array(group) for group in point_groups]

    return point_groups

def group_points_by_grid_optimized(point_cloud, side_length):
    """
    Group points from a 2D point cloud based on a grid defined by a given side length.

    Parameters:
    - point_cloud (numpy.ndarray): Array of shape (N, 2) representing the 2D coordinates of points.
    - side_length (float): Length of the side of each grid cell.

    Returns:
    - tuple: Tuple containing:
        - list: List of NumPy arrays containing groups of points.
        - dict: Dictionary mapping group indices to points.
        - dict: Dictionary mapping group indices to corresponding coordinates.

    This function divides the 2D point cloud into grid cells based on the specified side length,
    assigns each point to the corresponding cell, and returns a list of arrays, each containing
    the points belonging to a particular grid cell. It also returns dictionaries mapping group
    indices to points and coordinates.
    """

    # Extract x and y coordinates from the point cloud
    x_coords = point_cloud[:, 0]
    y_coords = point_cloud[:, 1]

    # Calculate the number of divisions in x and y directions
    num_x_divisions = int(np.ceil((x_coords.max() - x_coords.min()) / side_length))
    num_y_divisions = int(np.ceil((y_coords.max() - y_coords.min()) / side_length))

    # Calculate the grid indices for each point
    x_indices = ((x_coords - x_coords.min()) / side_length).astype(int)
    y_indices = ((y_coords - y_coords.min()) / side_length).astype(int)

    # Calculate the unique group indices for each point
    group_indices = y_indices * num_x_divisions + x_indices

    # Initialize a dictionary to store groups of points
    point_groups_dict = {index: [] for index in np.unique(group_indices)}

    # Assign each point to the corresponding group using vectorized indexing
    for i, group_index in tqdm(enumerate(group_indices), total=len(group_indices), leave=False):
        point_groups_dict[group_index].append(point_cloud[i])

    # Convert the dictionary values to NumPy arrays
    point_groups = [np.array(group) for group in point_groups_dict.values()]
    
    coords = dict(zip(group_indices, zip(x_coords, y_coords)))
    
    return point_groups, point_groups_dict, coords
    
def runModel(pc, resolution, model, batch_size, point_thresh, device):
    """
    Run a model on grouped and normalized point cloud data.

    Parameters:
    - pc (numpy.ndarray): Array of shape (N, 3) representing the 3D coordinates of points.
    - resolution (float): Resolution for grouping points by grid.
    - model: Trained model to be run on the data.
    - batch_size (int): Batch size for model inference.
    - point_thresh (int): Threshold for number of points in each group.
    - device: Device on which to run the model (e.g., "cpu" or "cuda").

    Returns:
    - tuple: Tuple containing:
        - numpy.ndarray: Outputs from the model.
        - dict: Dictionary mapping group indices to points.
        - dict: Dictionary mapping group indices to corresponding coordinates.

    This function first groups points from the input point cloud by grid, then normalizes
    each group by subtracting their means. It then prepares the data for model inference,
    runs the model on the data in batches, and returns the outputs along with dictionaries
    mapping group indices to points and coordinates.
    """
    
    # GROUP POINTS BY WHERE THEY ARE (THEIR GRID CELL)
    groups, group_mapping, coord_mapping = group_points_by_grid_optimized(pc, resolution)
    
    # NORMALIZE GROUPS BY THEIR MEANS 
    groups_normed = []
    for group in tqdm(groups, leave=False):
        test = group.copy()
        test[:, 0] = test[:, 0] - test[:, 0].mean()
        test[:, 1] = test[:, 1] - test[:, 1].mean()
        test[:, 2] = test[:, 2] - test[:, 2].min()
        
        # IF TOO MANY POINTS, THRESHOLD RANDOMLY
        if test.shape[0] > point_thresh:
            test = test[np.random.choice(test.shape[0], point_thresh, replace=False), :]

        groups_normed.append(test)
    
    # FORMAT FOR MODEL RUN
    groups_normed_tensor = [torch.Tensor(x) for x in groups_normed]
    padded_sequences = pad_sequence(groups_normed_tensor, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences.movedim(-1, 1)
    
    # RUN DNN
    outputs = torch.zeros(padded_sequences.shape[0], 1)
    for i in tqdm(np.arange(0, padded_sequences.shape[0], batch_size), leave=False):
        x = batch_size
        if i + x > padded_sequences.shape[0]:
            x = padded_sequences.shape[0]
        temp_out = model(padded_sequences[i:i+x].to(device))
        outputs[i:i+x] = temp_out.cpu().detach()
        torch.cuda.empty_cache()

    outputs = outputs.numpy() / 10000 # UNDO MODEL SCALING
    return outputs, group_mapping, coord_mapping

def filterGridByDistance(threshold, values, mesh_x, mesh_y, points_x, points_y):
    """
    Filter grid values by distance threshold from original points.

    Parameters:
    - threshold (float): Maximum distance threshold for inclusion.
    - values (numpy.ndarray): Array of grid values.
    - mesh_x (numpy.ndarray): X coordinates of grid.
    - mesh_y (numpy.ndarray): Y coordinates of grid.
    - points_x (numpy.ndarray): X coordinates of original points.
    - points_y (numpy.ndarray): Y coordinates of original points.

    Returns:
    - numpy.ndarray: Filtered grid values.

    This function filters grid values based on the distance threshold from original points.
    Grid cells containing points within the threshold distance are retained, while others are
    assigned NaN values.
    """

    # https://stackoverflow.com/questions/30655749/how-to-set-a-maximum-distance-between-points-for-interpolation-when-using-scipy
    
    query_points = np.vstack([mesh_x.flatten(), mesh_y.flatten()]).T
    
    # Construct kd-tree, functionality copied from scipy.interpolate
    tree = cKDTree(np.c_[points_x.ravel(), points_y.ravel()])
    dists, indexes = tree.query(query_points)
    
    dists = dists.reshape(values.shape)
    
    # Copy original result but mask missing values with NaNs
    result = values
    result[dists > threshold] = np.nan
    return result


def getModelGrid_LAS(lasdir, model_dir, 
                     ground_class=None,
                     resolution = 1, 
                     distance = 3,
                     point_thresh = 20,
                     batch_size = 1000,
                     diagnostics=False,
                     save=None,
                     epsg_string="EPSG:32615",
                     device='cuda'
                    ):


    """
    Process LAS point cloud data with a deep learning model and return the interpolated grid.

    Parameters:
    - lasdir (str): Path to the LAS file.
    - model_dir (str): Path to the trained model file.
    - ground_class (int or None, optional): Ground class label for filtering points. Default is None.
    - resolution (float, optional): Resolution for grid interpolation. Default is 1.
    - distance (float, optional): Distance threshold for filtering grid points. Default is 3.
    - point_thresh (int, optional): Threshold for the number of points in each group for inference. Default is 20.
    - batch_size (int, optional): Batch size for model inference. Default is 1000.
    - diagnostics (bool, optional): Whether to return additional diagnostic information. Default is True.
    - save (str or None, optional): Currently broken. Path to save the interpolated grid as a GeoTIFF file. Default is None.
    - epsg_string (str, optional): EPSG string for the coordinate reference system. Default is "EPSG:32615". 
    - device (str, optional): Device on which to run the model (e.g., 'cpu' or 'cuda'). Default is 'cuda'.

    Returns:
    - numpy.ndarray: Interpolated grid of the processed point cloud data.

    This function reads the LAS file, optionally filters points by ground class, runs the deep learning model on the
    point cloud data, interpolates the model outputs onto a grid, filters the grid by distance from original points,
    and optionally saves the result as a GeoTIFF file. If `diagnostics` is True, it also returns additional
    diagnostic information along with the interpolated grid.

    NOTE: EPSG_STRING implementation is currently broken. Needs to be fixed. 
    """

    # READ POINT CLOUD AND FORMAT AS NUMPY ARRAY
    las = laspy.read(lasdir)
    
    if ground_class is not None:
        new_file = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
        new_file.points = las.points[las.classification == ground_class]
        las = new_file
    
    pc = np.dstack((las.x, las.y, las.z)).squeeze()
    
    # LOAD MODEL
    model = torch.load(model_dir, map_location=torch.device(device))
    
    # RUN MODEL
    outputs, group_mapping, coord_mapping = runModel(pc, resolution, model, batch_size, point_thresh, device)
    
    # REMOVE MODEL FROM GPU AND CLEAR
    model = model.to("cpu")
    torch.cuda.empty_cache()
    
    # GET XY COORDINATES FOR EACH CELL (TODO: ADD CENTER COORD INSTEAD OF CORNER)
    coords = np.array([coord_mapping[x] for x in list(group_mapping.keys())])
    x = coords[:, 0]
    y = coords[:, 1]
    
    # INTERPOLATE GRID    
    grid_x, grid_y = np.mgrid[min(x):max(x):resolution, min(y):max(y):resolution]
    grid_z = griddata((x, y), outputs, (grid_x, grid_y), method='cubic')
    
    # FILTER BY DISTANCE
    grid_z = filterGridByDistance(distance, grid_z, grid_x, grid_y, pc[:, 0], pc[:, 1])
    
    if save is not None:
        transform = from_origin(min(x), max(y), resolution, resolution)
        metadata = {'driver': 'GTiff', 
                    'count': 1, 
                    'dtype': 'float64', 
                    'width': grid_z.shape[0], 
                    'height': grid_z.shape[1], 
                    # 'crs': epsg_string, 
                    'transform': transform}
        # Write the raster to GeoTIFF
        with rasterio.open(save, 'w', **metadata) as dst:
            dst.write(grid_z[:, ::-1, 0].T, 1)
    
    if diagnostics:
        return grid_z, (outputs, group_mapping, coord_mapping, coords)
    
    return grid_z