

class EncodedModelInputs:
    def __init__(self, position_encoder, direction_encoder, rays_from_camera_builder, point_sampler,
                 depth_samples_per_ray):
        super().__init__()
        self.pos_encoding_function = position_encoder
        self.dir_encoding_function = direction_encoder
        self.rays_from_camera_builder = rays_from_camera_builder
        self.point_sampler = point_sampler
        self.depth_samples_per_ray = depth_samples_per_ray

    def encoded_points_and_directions_from_camera(self, tform_cam2world):
        ray_origins, ray_directions = self.rays_from_camera_builder.ray_origins_and_directions_from_pose(
            tform_cam2world)

        query_points, depth_values = self.point_sampler.query_points_on_rays(ray_origins, ray_directions)

        ray_directions = expand_ray_directions_to_fit_ray_query_points(ray_directions, query_points)

        encoded_query_points = self.pos_encoding_function.forward(query_points)#query_points.reshape(-1, 3)).reshape((10000, 50, -1))
        encoded_ray_directions = self.dir_encoding_function.forward(ray_directions)

        return encoded_query_points, encoded_ray_directions, depth_values

def expand_ray_directions_to_fit_ray_query_points(ray_directions, query_points):
    ray_dir_new_shape = (query_points.shape[0], query_points.shape[1], 1, 3)
    ray_directions = ray_directions.reshape(ray_dir_new_shape).expand(query_points.shape)

    return ray_directions