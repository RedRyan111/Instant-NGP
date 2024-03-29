import torch
from tqdm import tqdm
from Instant_ngp.encode_model_inputs import EncodedModelInputs
from Instant_ngp.voxel_hashing import HashManager
from data_loaders.tiny_data_loader import DataLoader
# from data_loaders.lego_data_loader import DataLoader
from display_utils.display_helper import display_image, create_video
from Instant_ngp.models.full_model import NerfModel
from Instant_ngp.nerf_forward_pass import ModelIteratorOverRayChunks
from Instant_ngp.positional_encoding import PositionalEncoding
#from Instant_ngp.point_sampler_on_rays.nerf_point_sampler import PointSamplerFromRays
from Instant_ngp.point_sampler_on_rays.instant_ngp_point_sampler import PointSamplerFromRays
from Instant_ngp.rays_from_camera_builder import RaysFromCameraBuilder
from setup_utils import set_random_seeds, load_training_config_yaml, get_tensor_device

set_random_seeds()
training_config = load_training_config_yaml()
device = get_tensor_device()
#device = "cpu"
data_manager = DataLoader(device)

# training parameters
lr = training_config['training_variables']['learning_rate']
num_iters = training_config['training_variables']['num_iters']
num_directional_encoding_functions = training_config['positional_encoding']['num_directional_encoding_functions']
depth_samples_per_ray = training_config['rendering_variables']['depth_samples_per_ray']
chunksize = training_config['rendering_variables']['samples_per_model_forward_pass']
size = 10
resolutions = [i*1.5 for i in range(12, 40)]#[i*1.5 for i in range(3, 20)]##[3, 4, 5, 6, 7, 8]#[2, 4, 8, 16, 32, 64]
embedding_lengths = [9 for i in range(12, 40)]#[13 for i in range(3, 20)]##[10, 10, 10, 11, 11, 11]
num_positional_encoding_functions = sum(embedding_lengths) * depth_samples_per_ray
print(f'{len(resolutions)} embedding: {len(embedding_lengths)} sum: {sum(embedding_lengths)} num pos enc: {num_positional_encoding_functions}')

# Misc parameters
display_every = training_config['display_variables']['display_every']

# Specify encoding classes
position_encoder = HashManager(size, resolutions, embedding_lengths, device)#.to(device)
direction_encoder = PositionalEncoding(3, num_directional_encoding_functions, device, True)
collision_detection = nerfacc.OccGridEstimator(roi_aabb=[-5, -5, -5, 5, 5, 5], resolution=10, levels=1).to(device)

# Initialize model and optimizer
model = NerfModel(num_positional_encoding_functions, num_directional_encoding_functions).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Setup ray classes
point_sampler = PointSamplerFromRays(training_config)
rays_from_camera_builder = RaysFromCameraBuilder(data_manager, device)

encoded_model_inputs = EncodedModelInputs(position_encoder,
                                          direction_encoder,
                                          rays_from_camera_builder,
                                          point_sampler,
                                          depth_samples_per_ray)

psnrs = []
for i in tqdm(range(num_iters)):

    target_img, target_tform_cam2world = data_manager.get_image_and_pose(i)

    #encoded_points_on_ray, encoded_ray_directions, depth_values = encoded_model_inputs.encoded_points_and_directions_from_camera(target_tform_cam2world)

    ray_origins, ray_directions = rays_from_camera_builder.ray_origins_and_directions_from_pose(target_tform_cam2world)
    print(f'ray origins: {ray_origins.shape} ray directions: {ray_directions.shape}')

    #ray_indices, t_starts, t_ends = estimator.sampling(ray_origins, ray_directions)
    print(f'ray_indecis: {ray_indices.shape} t_starts: {t_starts.shape} t_ends: {t_ends.shape}')

    #query_points, depth_values = point_sampler.query_points_on_rays(ray_origins, ray_directions) custom point sampler

    model_forward_iterator = ModelIteratorOverRayChunks(chunksize, encoded_points_on_ray, encoded_ray_directions, depth_values,
                                                        target_img, model)

    predicted_image = []
    loss_sum = 0
    for predicted_pixels, target_pixels in model_forward_iterator:
        loss = torch.nn.functional.mse_loss(predicted_pixels, target_pixels)
        loss.backward()
        loss_sum += loss.detach()

        predicted_image.append(predicted_pixels)

    optimizer.step()
    optimizer.zero_grad()
    model.zero_grad()

    predicted_image = torch.concatenate(predicted_image, dim=0).reshape(target_img.shape[0], target_img.shape[1], 3)

    if i % display_every == 0:
        psnr = -10. * torch.log10(loss_sum)
        psnrs.append(psnr.item())

        print("Loss:", loss_sum)
        display_image(i, display_every, psnrs, predicted_image, target_img)

    if i == num_iters - 1:
        #save_image(display_every, psnrs, predicted_image)
        create_video()

print('Done!')
