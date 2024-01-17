import torch
import torch.nn as nn
from tqdm import tqdm

from Instant_ngp.occupancy_grid import OccupancyManager
from Instant_ngp.rays_from_camera_builder import RaysFromCameraBuilder
from data_loaders.tiny_data_loader import DataLoader
from setup_utils import get_tensor_device

device = get_tensor_device()
data_manager = DataLoader(device)
rays_from_camera_builder = RaysFromCameraBuilder(data_manager, device)

num_iters = 1
for i in tqdm(range(num_iters)):
    target_img, target_tform_cam2world = data_manager.get_image_and_pose(i)

    ray_origins, ray_directions = rays_from_camera_builder.ray_origins_and_directions_from_pose(target_tform_cam2world)

    print(f'ray origins: {ray_origins.shape} ray directions: {ray_directions.shape}')

size = 3
resolution = 2
embedding_length = 1

voxel_hash = OccupancyManager(size, resolution, embedding_length, device)



xyz_tensor = torch.cat([torch.zeros(1, 3)-1.49, torch.zeros(1, 3)+1.49], dim=0)
print(f'xyz tensor: ')
print(xyz_tensor.shape)

#corner_embeddings = voxel_hash.get_corner_embedding_vectors(xyz_tensor)

print(f'starting!')

#print(f'embedding: {corner_embeddings}')


xyz_embeddings = voxel_hash.get_embedding(xyz_tensor)

print(xyz_embeddings)
