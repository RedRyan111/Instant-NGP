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
        self.resolution = int(resolution)
        self.testable_range = size * ((torch.arange(self.resolution + 1) / self.resolution) - .5)
        self.testable_range = self.testable_range.to(device)
        self.stacked_range = self.testable_range.repeat(3)
        self.grid_size = size / resolution
        self.device = device
        self.indicis = self.get_indecis()


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
        print(f'stacked range: {self.stacked_range.shape}')

        print(f'position: {position.shape} direction: {direction.shape}')

        t = (self.stacked_range - position) / direction
        print(f'div: {t.shape}')
        print(t[0:2])

        return t


    def calc_t(self, stacked_range, position, direction):
        return (self.stacked_range - position) / direction


    def get_indecis(self):
        index_size = (self.resolution - 1) ** 3
        max_x, max_y, max_z = [], [], []
        min_x, min_y, min_z = [], [], []

        for i in range(self.resolution):
            for j in range(self.resolution):
                for k in range(self.resolution):
                    max_x.append(i)
                    max_y.append(j)
                    max_z.append(k)

                    min_x.append(i+1)
                    min_y.append(j+1)
                    min_z.append(k+1)

        print(f'first max_x shape: {torch.tensor(max_x).shape}')

        t0_x = torch.tensor(max_x).to(self.device)
        t0_y = torch.tensor(max_y).to(self.device)
        t0_z = torch.tensor(max_z).to(self.device)

        t1_x = torch.tensor(min_x).to(self.device)
        t1_y = torch.tensor(min_y).to(self.device)
        t1_z = torch.tensor(min_z).to(self.device)

        return t0_x, t0_y, t0_z, t1_x, t1_y, t1_z

    def sample(self, position, direction):
        stacked_range = self.testable_range.repeat(*position.shape).reshape(position.shape[0], 3, -1)
        print(f'stacked range print: ')
        print(stacked_range[0:2])
        print(f'position print: ')
        print(position[0:2])
        print(f'direction print: ')
        print(direction[0:2])
        t = (stacked_range - position.unsqueeze(2)) / direction.unsqueeze(2)
        print(f't shape: {t.shape}')
        print('final t:')
        print(t[0:2])
        print(f'stacked range: {stacked_range.shape} position: {position.shape} direction: {direction.shape} sliced direction: {direction[:, 0].shape}')
        print(f't: {t.shape}')

        #expand direction?



        #t = (self.testable_range - position) /direction

        t0_x, t0_y, t0_z, t1_x, t1_y, t1_z = self.get_indecis()
        print(f't0_x: {t0_x.shape}')

        direction = direction.unsqueeze(dim=2).repeat(1, 1, t0_x.shape[0])
        print(f'direction: {direction.shape}')

        t0_x = torch.where(direction[:, 0] >= 0, t0_x, t1_x)
        t1_x = torch.where(direction[:, 0] < 0, t0_x, t1_x)

        t0_y = torch.where(direction[:, 1] >= 0, t0_x, t1_x)
        t1_y = torch.where(direction[:, 1] < 0, t0_y, t1_y)

        t0_z = torch.where(direction[:, 2] >= 0, t0_z, t1_z)
        t1_z = torch.where(direction[:, 2] < 0, t0_z, t1_z)
        print(f't0_x: {t0_x.shape}')

        t0_xyz = torch.stack([t0_x, t0_y, t0_z], dim=2)
        t1_xyz = torch.stack([t1_x, t1_y, t1_z], dim=2)

        #these are indices, not t values
        print(f'full shape: {t0_xyz.shape}')

        t_min, _ = torch.min(t0_xyz, dim=2, keepdim=True)
        t_max, _ = torch.min(t1_xyz, dim=2, keepdim=True)

        print(f't_min: {t_min.shape} t_max: {t_max.shape}')
        print(f' t min: ')
        print(t_min.squeeze())
        print(f' t max: ')
        print(t_max.squeeze())

        bool_tensor = (t_max > t_min) * (t_max > 0) #t_max > 0 prevents hits behind ray origin
        #bool_tensor = torch.where(t_max > t_min and t_max > 0, 1, 0)
        bool_tensor = bool_tensor.squeeze()

        print(f'bool tensor: {bool_tensor.shape}')
        print(f'final bools:')
        print(bool_tensor)

        #get indecis to sample
        #sample indecis
        #delete duplicates and then sample?
        #weight number of samples based on the difference between t_max and t_min?

        return 0

    def forward(self, xyz):
        embeddings = []
        for hasher in self.hash_list:
            embeddings.append(hasher(xyz))

        return torch.cat(embeddings, dim=-1)
