import torch

from Instant_ngp.voxel_hashing import VoxelHash


size = 3
resolution = 2
embedding_length = 1

voxel_hash = VoxelHash(size, resolution, embedding_length)

#xyz_tensor = torch.tensor([[.75, 1, 1.25]])
xyz_tensor = torch.rand((2, 3))
print(f'xyz tensor: ')
print(xyz_tensor)

embedding = voxel_hash.get_embedding(xyz_tensor)

print(f'starting!')

print(f'embedding: {embedding}')
