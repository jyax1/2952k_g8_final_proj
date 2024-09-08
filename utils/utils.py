from datasets import *
from architectures.direct_cnn_mlp import ActionExtractionCNN
from architectures.direct_cnn_vit import ActionExtractionViT
from architectures.latent_cnn_unet import ActionExtractionCNNUNet
from architectures.direct_resnet_mlp import ActionExtractionResNet
from architectures.latent_decoders import *
import re
from pathlib import Path

def center_crop(tensor, output_size=112):
    '''
    Temporary function to match input size of DINOv2
    '''
    # Ensure the input tensor is of shape [n, 3, 128, 128]
    assert tensor.ndim == 4 and tensor.shape[1] == 3 and tensor.shape[2] == 128 and tensor.shape[3] == 128, \
    "Input tensor must be of shape [n, 3, 128, 128]"

    # Calculate the starting points for the crop
    crop_size = output_size
    h_start = (tensor.shape[2] - crop_size) // 2
    w_start = (tensor.shape[3] - crop_size) // 2

    # Perform the crop
    cropped_tensor = tensor[:, :, h_start:h_start + crop_size, w_start:w_start + crop_size]
    
    return cropped_tensor


def load_datasets(
        architecture, 
        datasets_path, 
        train=True, 
        validation=True, 
        horizon=2, 
        demo_percentage=0.9, 
        cameras=['frontview_image'],
        motion=False,
        image_plus_motion=False
):
    if 'latent' in architecture and 'decoder' not in architecture and 'aux' not in architecture:
        if train:
            train_set = DatasetVideo(path=datasets_path, x_pattern=[0,1], y_pattern=[1],
                                            demo_percentage=demo_percentage, cameras=cameras)
        if validation:
            validation_set = DatasetVideo(path=datasets_path, x_pattern=[0,1], y_pattern=[1],
                                                    demo_percentage=.9, cameras=cameras, validation=True)
    elif 'latent' in architecture and 'aux' in architecture:
        if train:
            train_set = DatasetVideo2VideoAndAction(path=datasets_path, x_pattern=[0,1], y_pattern=[1],
                                            demo_percentage=demo_percentage, cameras=cameras)
        if validation:
            validation_set = DatasetVideo2VideoAndAction(path=datasets_path, x_pattern=[0,1], y_pattern=[1],
                                                    demo_percentage=.9, cameras=cameras, validation=True)
    else:
        if train:
            train_set = DatasetVideo2DeltaAction(path=datasets_path, video_length=horizon, 
                                            demo_percentage=demo_percentage, cameras=cameras,
                                            motion=motion, image_plus_motion=image_plus_motion)
        if validation:
            validation_set = DatasetVideo2DeltaAction(path=datasets_path, video_length=horizon, 
                                                demo_percentage=demo_percentage, cameras=cameras, validation=True, 
                                                motion=motion, image_plus_motion=image_plus_motion)

    if train and validation:
        return train_set, validation_set
    elif train and not validation:
        return train_set
    elif validation and not train:
        return validation_set
    
def load_model(architecture, 
               horizon, 
               results_path, 
               latent_dim, 
               motion, 
               image_plus_motion, 
               vit_patch_size, 
               resnet_layers_num,
               idm_model_name,
               fdm_model_name,
               freeze_idm,
               freeze_fdm,
               dinov2
):
    if architecture == 'direct_cnn_mlp':
        model = ActionExtractionCNN(latent_dim=latent_dim, 
                                    video_length=horizon, 
                                    motion=motion, 
                                    image_plus_motion=image_plus_motion)
    elif architecture == 'direct_cnn_vit':
        model = ActionExtractionViT(latent_dim=latent_dim, 
                                    video_length=horizon, 
                                    motion=motion, 
                                    image_plus_motion=image_plus_motion,
                                    vit_patch_size=vit_patch_size)
    elif architecture == 'direct_resnet_mlp':
        resnet_version = 'resnet' + str(resnet_layers_num)
        model = ActionExtractionResNet(resnet_version, action_length=horizon-1, num_mlp_layers=3, dinov2=dinov2)
    elif architecture == 'latent_cnn_unet':
        model = ActionExtractionCNNUNet(latent_dim=latent_dim, video_length=horizon) # doesn't support motion
    elif 'latent_decoder' in architecture:
        idm_model_path = str(Path(results_path)) + f'/{idm_model_name}'
        latent_dim = int(re.search(r'_lat(.*?)_', idm_model_name).group(1))
        if architecture == 'latent_decoder_mlp':
            model = LatentDecoderMLP(idm_model_path, 
                                     latent_dim=latent_dim, 
                                     video_length=horizon, 
                                     latent_length=horizon-1, 
                                     mlp_layers=10)
        elif architecture == 'latent_decoder_vit':
            model = LatentDecoderTransformer(idm_model_path, 
                                             latent_dim=latent_dim, 
                                             video_length=horizon, 
                                             latent_length=horizon-1,
                                             vit_patch_size=vit_patch_size)
        elif architecture == 'latent_decoder_obs_conditioned_unet_mlp':
            model = LatentDecoderObsConditionedUNetMLP(idm_model_path,
                                                       latent_dim=latent_dim,
                                                       video_length=horizon,
                                                       latent_length=horizon-1,
                                                       mlp_layers=10)
        elif architecture == 'latent_decoder_aux_separate_unet_vit':
            fdm_model_path = str(Path(results_path)) + f'/{fdm_model_name}'
            model = LatentDecoderAuxiliarySeparateUNetTransformer(idm_model_path, 
                                                        fdm_model_path, 
                                                        latent_dim=latent_dim, 
                                                        video_length=horizon, 
                                                        freeze_idm=freeze_idm, 
                                                        freeze_fdm=freeze_fdm,
                                                        vit_patch_size=vit_patch_size)
        elif architecture == 'latent_decoder_aux_separate_unet_mlp':
            fdm_model_path = str(Path(results_path)) + f'/{fdm_model_name}'
            model = LatentDecoderAuxiliarySeparateUNetMLP(idm_model_path, 
                                                        fdm_model_path, 
                                                        latent_dim=latent_dim, 
                                                        video_length=horizon, 
                                                        freeze_idm=freeze_idm, 
                                                        freeze_fdm=freeze_fdm)
        elif architecture == 'latent_decoder_aux_combined_vit':
            fdm_model_path = str(Path(results_path)) + f'/{fdm_model_name}'
            model = LatentDecoderAuxiliaryCombinedViT(idm_model_path, 
                                                        latent_dim=latent_dim, 
                                                        video_length=horizon, 
                                                        freeze_idm=freeze_idm)

    return model