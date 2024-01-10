import torch
import torch.nn as nn
from Instant_ngp.voxel_hashing import VoxelHash


size = 3
resolution = 2
embedding_length = 1

num_of_embeddings = (resolution + 1) ** 3
embedding = nn.Embedding(num_of_embeddings, embedding_length)
for i in range(num_of_embeddings):
    with torch.no_grad():
        embedding.weight[i] = i * torch.ones(embedding_length)
    print(f'index: {i} weight: {embedding.weight[i]}')


voxel_hash = VoxelHash(size, resolution, embedding_length)
voxel_hash.embedding = embedding

#xyz_tensor = torch.tensor([[.75, 1, 1.25]])
xyz_tensor = torch.rand((2, 3))
print(f'xyz tensor: ')
print(xyz_tensor)

#embedding = voxel_hash.get_embedding(xyz_tensor)
corner_embeddings = voxel_hash.get_corner_embedding_vectors(xyz_tensor)

print(f'starting!')

print(f'embedding: {corner_embeddings}')
