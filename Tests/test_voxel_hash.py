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

xyz_tensor = torch.cat([torch.zeros(1, 3)-1.49, torch.zeros(1, 3)+1.49], dim=0)
print(f'xyz tensor: ')
print(xyz_tensor.shape)

#corner_embeddings = voxel_hash.get_corner_embedding_vectors(xyz_tensor)

print(f'starting!')

#print(f'embedding: {corner_embeddings}')


xyz_embeddings = voxel_hash.get_embedding(xyz_tensor)

#rand_tensor = torch.rand((2, 8, 2))
#temp_1 = torch.rand((2, 2)).unsqueeze(1).expand_as(rand_tensor)
#print(f'temp 1 shape: {temp_1.shape}')

#print(rand_tensor - temp_1)

'''
cube_of_indecis = torch.stack([
    torch.stack([torch.ones(2), torch.ones(2)*2, torch.ones(2)*3], dim=1),
    torch.stack([torch.ones(2), torch.ones(2)*2, torch.ones(2)*3], dim=1),
    torch.stack([torch.ones(2), torch.ones(2)*2, torch.ones(2)*3], dim=1)
], dim=1)

print(cube_of_indecis)
print(f'cube shape: {cube_of_indecis.shape}')
print(f'only x:')
print(cube_of_indecis[:, :, 0])
'''