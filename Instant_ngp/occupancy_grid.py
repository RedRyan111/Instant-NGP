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


class OccupancyManager(nn.Module):
    def __init__(self, size, resolution, embedding_length, device):
        super().__init__()
        self.occupancy_hash = VoxelHash(size, resolution, embedding_length).to(device)
        self.resolution = resolution
        self.testable_range = size * ((torch.arange(self.resolution + 1) / self.resolution) - .5)
        self.testable_range = self.testable_range.to(device)
        self.stacked_range = self.testable_range.repeat(3)
        self.grid_size = size / resolution

    def fit_tensor_to_range(self, inp_tensor):
        inp_tensor = inp_tensor.reshape((-1, 3))
        inp_tensor = torch.repeat_interleave(inp_tensor, self.resolution+1, dim=1)
        return inp_tensor

    def get_all_t_boundaries_from_rays(self, position, direction):
        position = self.fit_tensor_to_range(position)
        print(f'position: -----------------------------------')
        print(position[0:2])

        direction = self.fit_tensor_to_range(direction)
        print(f'direction: -----------------------------------')
        print(direction[0:2])

        print(f'stacked range: {self.stacked_range.shape}')
        print(f'grid size {self.grid_size}')
        print(f'stacked range: {self.stacked_range}')

        print(f'position: {position.shape} direction: {direction.shape}')

        t = (self.stacked_range - position) / direction
        print(f'div: {t.shape}')
        print(t[0:2])

        return t

    def sample(self, position, direction):
        t = self.get_all_t_boundaries_from_rays(position, direction)
        print(f't: {t.shape}')

        #might be able to get rid of permute with earlier shape optimizations
        #t = t.reshape(-1, 3, self.resolution+1).permute((0, 2, 1)) #might have to permute, not sure yet
        t = t.reshape(-1, 3, self.resolution + 1)#.permute((0, 2, 1))
        print(f't: {t.shape}')

        print(f'slices: {t[:, :, 0]}')
        print(f'slice shape: {t[:, :, 0].shape}')

        maximums = torch.max(t[:, :, 0], dim=1)
        print(f'maxs: {maximums[0:2]}')

        minimums = torch.min(t[:, :, -1], dim=1)
        print(f'mins: {minimums[0:2]}')

        #t = t + self.grid_size / 2  # small enough to not go to neighboring bounding boxes, gets inbetween indecis
        #min_t = torch.where(t > -1 * self.grid_size, t, -1*self.grid_size) #this doent work, its t not grid size
        #valid_t = torch.where(min_t < self.grid_size, min_t, self.grid_size)

        return 0

    def forward(self, xyz):
        embeddings = []
        for hasher in self.hash_list:
            embeddings.append(hasher(xyz))

        return torch.cat(embeddings, dim=-1)
