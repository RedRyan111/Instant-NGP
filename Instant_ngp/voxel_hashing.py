import torch
import torch.nn as nn
import math
from itertools import permutations


class HashManager(nn.Module):
    def __init__(self, number_of_hashes):
        super().__init__()

        hash_list = [VoxelHash() for i in range(number_of_hashes)]


class VoxelHash(nn.Module):
    def __init__(self, size, resolution, embedding_length):
        super().__init__()
        self.size = size
        self.resolution = resolution
        self.embedding_length = embedding_length

        num_of_hashes_per_dimension = size / resolution

        num_of_embeddings = (resolution+1) ** 3
        print(f'number of embeddings: {num_of_embeddings}')

        self.embedding = nn.Embedding(num_of_embeddings, embedding_length)

    #def is_in_bounds(self, x, y, z):
    #    if x > self.size or y > self.size or z > self.size:
    #        raise

    def linear_interpolation(self):
        return 0

    def get_embedding(self, xyz_tensor):
        corner_embeddings = self.get_corner_embedding_vectors(xyz_tensor)

        # trilinearly interpolate

        return 0

    def get_corner_embedding_vectors(self, xyz_tensor):
        multiplier = self.resolution / self.size
        xyz_tensor = multiplier * xyz_tensor + self.resolution/2

        print(f'xyz tensor shape: {xyz_tensor.shape}')

        xyz_floor = torch.floor(xyz_tensor)
        xyz_ceil = torch.ceil(xyz_tensor)

        bot_x_index = xyz_floor[:, 0]
        top_x_index = xyz_ceil[:, 0]
        bot_y_index = xyz_floor[:, 1] * self.size
        top_y_index = xyz_ceil[:, 1] * self.size
        bot_z_index = xyz_floor[:, 2] * self.size**2
        top_z_index = xyz_ceil[:, 2] * self.size**2

        print(f'bot x: {bot_x_index.shape}')

        temp = torch.stack([bot_x_index, bot_y_index, bot_z_index], dim=1)

        print(f'temp shape: {temp.shape}')

        cube_of_indecis = torch.stack([
            torch.stack([bot_x_index, bot_y_index, bot_z_index], dim=1),
            torch.stack([bot_x_index, bot_y_index, top_z_index], dim=1),
            torch.stack([bot_x_index, top_y_index, bot_z_index], dim=1),
            torch.stack([bot_x_index, top_y_index, top_z_index], dim=1),
            torch.stack([top_x_index, bot_y_index, bot_z_index], dim=1),
            torch.stack([top_x_index, bot_y_index, top_z_index], dim=1),
            torch.stack([top_x_index, top_y_index, bot_z_index], dim=1),
            torch.stack([top_x_index, top_y_index, top_z_index], dim=1)
        ], dim=1)

        print(f'cube of indecis: {cube_of_indecis.shape}')

        indecis = torch.sum(cube_of_indecis, dim=2, dtype=torch.int)#self.xyz_indecis_to_embedding_indecis(cube_of_indecis)

        print(f'indecis: {indecis.shape}')

        embeddings = self.embedding(indecis)
        print(f'embeddings: {embeddings.shape}')
        #embedding = torch.cat(embeddings, dim=1)

        return embeddings
