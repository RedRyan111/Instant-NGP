import torch

from Instant_ngp.render_volume_density import render_volume_density


class ModelIteratorOverRayChunks(object):
    def __init__(self, chunk_size, encoded_query_points, encoded_ray_directions, depth_values, target_image, model):
        self.chunk_size = chunk_size
        self.chunk_index = -1
        self.num_of_rays = encoded_ray_directions.shape[0]

        self.encoded_query_points = torch.split(encoded_query_points, chunk_size)
        self.encoded_ray_directions = torch.split(encoded_ray_directions, chunk_size)

        self.depth_values = torch.split(depth_values.reshape(self.num_of_rays, -1), chunk_size)
        self.target_image = torch.split(target_image.reshape(-1, 3), chunk_size)

        self.model = model

    def __iter__(self):
        return self

    def is_out_of_bounds(self):
        return (self.chunk_index + 1) * self.chunk_size >= self.num_of_rays

    def __next__(self):
        if self.is_out_of_bounds():
            raise StopIteration

        self.chunk_index += 1
        #replace encoded points with embedding algorithm
        encoded_points = self.encoded_query_points[self.chunk_index]
        encoded_ray_origins = self.encoded_ray_directions[self.chunk_index]

        depth_values = self.depth_values[self.chunk_index]

        rgb, density = self.model(encoded_points, encoded_ray_origins)

        rgb_predicted = render_volume_density(rgb, density, depth_values)

        return rgb_predicted, self.target_image[self.chunk_index]
