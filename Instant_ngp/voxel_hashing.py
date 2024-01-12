import torch
import torch.nn as nn


class HashManager(nn.Module):
    def __init__(self, size, resolutions, embedding_lengths, device):
        super().__init__()
        self.number_of_hashes = len(resolutions)

        self.hash_list = [VoxelHash(size, resolutions[i], embedding_lengths[i]).to(device) for i in range(self.number_of_hashes)]

    def forward(self, xyz):
        embeddings = []
        for hasher in self.hash_list:
            embeddings.append(hasher(xyz))

        return torch.cat(embeddings, dim=-1)


class VoxelHash(nn.Module):
    def __init__(self, size, resolution, embedding_length):
        super().__init__()
        self.size = size
        self.resolution = resolution
        self.embedding_length = embedding_length

        num_of_hashes_per_dimension = size / resolution

        num_of_embeddings = (resolution+1) ** 3
        #print(f'number of embeddings: {num_of_embeddings}')

        self.embedding = nn.Embedding(num_of_embeddings, embedding_length)

    def is_in_bounds(self, xyz_tensor):
        greater_than_size = torch.sum(torch.abs(xyz_tensor) > self.size/2, dim=-1)
        indecis = greater_than_size.nonzero().reshape(-1)
        if indecis.shape[0] != 0:
            raise Exception(f'Embedding is out of bounds!!: {indecis}')

    def is_in_indecis(self, xyz_tensor):
        return 0

    #get embeddings
    def forward(self, xyz_tensor):
        self.is_in_bounds(xyz_tensor)

        normalized_xyz_tensor = self.normalize_xyz(xyz_tensor)

        cube_of_xyz_coords = self.get_cube_of_xyz_coords(xyz_tensor)

        corner_embeddings = self.get_corner_embeddings_from_cube_of_xyz_coords(cube_of_xyz_coords)
        #print(f'corner_embeddings: {corner_embeddings.shape}')

        #trilinearyly interpolate corner embeddings and xyz_tensor

        normalized_xyz_tensor = normalized_xyz_tensor.unsqueeze(1).expand_as(cube_of_xyz_coords)
        #print(f'corrected xyz: {normalized_xyz_tensor.shape}')

        sub_vectors = (cube_of_xyz_coords - normalized_xyz_tensor)**2
        distance_vectors = torch.sqrt(torch.sum(sub_vectors, dim=2)) #maybe square root isn't neccessary?
        #print(f'distance vectors: {distance_vectors.shape}')

        sum_of_distances = torch.sum(distance_vectors, dim=1).reshape((-1, 1)).expand_as(distance_vectors)
        #print(f'sum of distances: {sum_of_distances.shape}')

        normalized_distance = distance_vectors / sum_of_distances
        #print(f'normalized distance vectors: {normalized_distance.shape}')

        final_embeddings = torch.sum(normalized_distance.unsqueeze(2) * corner_embeddings, dim=1) #weighted sum
        #print(f'final embedding: {final_embeddings.shape}')

        return final_embeddings

    def get_cube_of_xyz_coords(self, xyz_tensor):
        xyz_tensor = self.normalize_xyz(xyz_tensor)
        #print(f'xyz tensor shape: {xyz_tensor.shape}')

        xyz_floor = torch.floor(xyz_tensor)
        xyz_ceil = torch.ceil(xyz_tensor)

        bot_x_index = xyz_floor[:, 0]
        top_x_index = xyz_ceil[:, 0]
        bot_y_index = xyz_floor[:, 1]
        top_y_index = xyz_ceil[:, 1]
        bot_z_index = xyz_floor[:, 2]
        top_z_index = xyz_ceil[:, 2]

        #need to use this cube to get the weights for trilinear interpolation
        cube_of_xyz_coords = torch.stack([
            torch.stack([bot_x_index, bot_y_index, bot_z_index], dim=1),
            torch.stack([bot_x_index, bot_y_index, top_z_index], dim=1),
            torch.stack([bot_x_index, top_y_index, bot_z_index], dim=1),
            torch.stack([bot_x_index, top_y_index, top_z_index], dim=1),
            torch.stack([top_x_index, bot_y_index, bot_z_index], dim=1),
            torch.stack([top_x_index, bot_y_index, top_z_index], dim=1),
            torch.stack([top_x_index, top_y_index, bot_z_index], dim=1),
            torch.stack([top_x_index, top_y_index, top_z_index], dim=1)
        ], dim=1)

        return cube_of_xyz_coords

    def normalize_xyz(self, xyz_tensor):
        multiplier = self.resolution / self.size
        return multiplier * xyz_tensor + self.resolution / 2

    def get_corner_embeddings_from_cube_of_xyz_coords(self, cube_of_xyz_coords):

        #print(f'cube of indecis for embeddings: {cube_of_xyz_coords.shape}')

        cube_of_xyz_coords[:, :, 1] = cube_of_xyz_coords[:, :, 1] * self.resolution
        cube_of_xyz_coords[:, :, 2] = cube_of_xyz_coords[:, :, 2] * self.resolution ** 2

        indecis = torch.sum(cube_of_xyz_coords, dim=2, dtype=torch.int)

        #print(f'max indecis: {torch.max(indecis)} max possible: {self.resolution**3} resolution: {self.resolution}')
        #argmax = torch.argmax(indecis)
        #print(cube_of_xyz_coords[argmax])
        #print(indecis)
        embeddings = self.embedding(indecis)

        return embeddings
