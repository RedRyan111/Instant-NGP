from typing import Tuple
import torch
from torch import Tensor
from Instant_ngp.render_volume_density import render_volume_density
import nerfacc


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

        self.estimator = nerfacc.OccGridEstimator(roi_aabb=[-5, -5, -5, 5, 5, 5], resolution=10, levels=1)

        #estimator.train()
        # update occupancy grid
        #estimator.update_every_n_steps(
        #    step=step,
        #    occ_eval_fn=occ_eval_fn,
        #    occ_thre=1e-2,
        #)
        #estimator.eval()

    def __iter__(self):
        return self

    def is_out_of_bounds(self):
        return (self.chunk_index + 1) * self.chunk_size >= self.num_of_rays

    def __next__(self):
        if self.is_out_of_bounds():
            raise StopIteration

        self.chunk_index += 1
        # replace encoded points with embedding algorithm
        encoded_points = self.encoded_query_points[self.chunk_index]
        encoded_ray_directions = self.encoded_ray_directions[self.chunk_index].reshape(encoded_points.shape[0], -1)

        depth_values = self.depth_values[self.chunk_index]

        print(f'points: {encoded_points.shape} origins: {encoded_ray_directions.shape} depth: {depth_values.shape}')

        rgb, density = self.model(encoded_points, encoded_ray_directions)



        rgb_predicted = render_volume_density(rgb, density, depth_values)

        return rgb_predicted, self.target_image[self.chunk_index]

'''
        ray_indices, t_starts, t_ends = estimator.sampling(
            rays_o,
            rays_d,
            sigma_fn=sigma_fn,
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            stratified=radiance_field.training,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )

                    rgb, acc, depth, _ = render_image_with_occgrid(
                        radiance_field,
                        estimator,
                        rays,
                        # rendering options
                        near_plane=near_plane,
                        render_step_size=render_step_size,
                        render_bkgd=render_bkgd,
                        cone_angle=cone_angle,
                        alpha_thre=alpha_thre,
                    )
                    
                    
    def sigma_fn(self, rays_o, rays_d, t_starts: Tensor, t_ends: Tensor, ray_indices: Tensor) -> Tensor:
        """ Define how to query density for the estimator."""
        t_origins = rays_o[ray_indices]  # (n_samples, 3)
        t_dirs = rays_d[ray_indices]  # (n_samples, 3)
        positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
        sigmas = radiance_field.query_density(positions)
        return sigmas  # (n_samples,)

    def rgb_sigma_fn(self, rays_o, rays_d, t_starts: Tensor, t_ends: Tensor, ray_indices: Tensor) -> Tuple[
        Tensor, Tensor]:
        """ Query rgb and density values from a user-defined radiance field. """
        t_origins = rays_o[ray_indices]  # (n_samples, 3)
        t_dirs = rays_d[ray_indices]  # (n_samples, 3)
        positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
        rgbs, sigmas = radiance_field(positions, condition=t_dirs)
        return rgbs, sigmas  # (n_samples, 3), (n_samples,)
'''
