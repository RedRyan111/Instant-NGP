import math
import torch
import torch.nn as nn
from Instant_ngp.voxel_hashing import VoxelHash


def calculate_intersections(point):
    # Calculate intersections with grid planes along each axis
    intersection_x = calculate_intersection_with_plane(point, axis='x')
    intersection_y = calculate_intersection_with_plane(point, axis='y')
    intersection_z = calculate_intersection_with_plane(point, axis='z')

    return [intersection_x, intersection_y, intersection_z]


def calculate_intersection_with_plane(point, axis):
    return 0


class NewRays:
    def __init__(self, position):
        position = position.reshape(-1, 3)
        self.shape = position.shape
        self.x = position[:, 0]
        self.y = position[:, 1]
        self.z = position[:, 2]

    def __shape__(self):
        return self.shape

class OccupancyManager(nn.Module):
    def __init__(self, size, resolution, embedding_length, device):
        super().__init__()
        self.occupancy_hash = VoxelHash(size, resolution, embedding_length).to(device)
        self.resolution = resolution
        self.testable_range = size * ((torch.arange(self.resolution + 1) / self.resolution) - .5)
        self.testable_range = self.testable_range.to(device)
        self.stacked_range = self.testable_range.repeat(3)
        self.grid_size = size / resolution

    def sample(self, position, direction):
        #position = NewRays(position)
        #direction = NewRays(direction)
        # testable_range = #.repeat() #set range scale
        position = position.reshape((-1, 3))
        position = torch.repeat_interleave(position, 3, dim=1)
        print(f'position: -----------------------------------')
        print(position[0:2])

        direction = direction.reshape((-1, 3))
        direction = torch.repeat_interleave(direction, 3, dim=1)
        print(f'direction: -----------------------------------')
        print(direction[0:2])

        print(f'stacked range: {self.stacked_range.shape}')
        print(f'grid size {self.grid_size}')
        print(f'stacked range: {self.stacked_range}')

        print(f'position: {position.shape} direction: {direction.shape}')
        diff = self.stacked_range - position
        print(f'diff: {diff.shape}')

        #print(f'repeated: {stacked_x[0:2]}')

        # get parametric equations?

        return 0

    def forward(self, xyz):
        embeddings = []
        for hasher in self.hash_list:
            embeddings.append(hasher(xyz))

        return torch.cat(embeddings, dim=-1)


def solve_for_t_in_parametric_equations():
    return 0
