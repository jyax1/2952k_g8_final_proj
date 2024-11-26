from datasets import *
from action_extractor.architectures.direct_cnn_mlp import ActionExtractionCNN, PoseExtractionCNN3D
from action_extractor.architectures.direct_cnn_vit import ActionExtractionViT
from action_extractor.architectures.latent_encoders import LatentEncoderPretrainCNNUNet, LatentEncoderPretrainResNetUNet
from action_extractor.architectures.direct_resnet_mlp import *
from action_extractor.architectures.latent_decoders import *
from action_extractor.architectures.resnet import *
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
        valsets_path,
        train=True, 
        validation=True, 
        horizon=2, 
        demo_percentage=0.9, 
        num_demo_train=5000,
        val_demo_percentage=0.9,
        cameras=['frontview_image'],
        motion=False,
        image_plus_motion=False,
        action_type='',
        data_modality='rgb',
        compute_stats=True,
        coordinate_system='disentangled'
):
    if 'latent' in architecture and 'decoder' not in architecture and 'aux' not in architecture:
        if train:
            train_set = DatasetVideo(path=datasets_path, x_pattern=[0,1], y_pattern=[1],
                                            demo_percentage=demo_percentage, cameras=cameras)
        if validation:
            validation_set = DatasetVideo(path=valsets_path, x_pattern=[0,1], y_pattern=[1],
                                                    demo_percentage=.9, cameras=cameras, validation=True)
    elif 'latent' in architecture and 'aux' in architecture:
        if train:
            train_set = DatasetVideo2VideoAndAction(path=datasets_path, x_pattern=[0,1], y_pattern=[1],
                                            demo_percentage=demo_percentage, cameras=cameras)
        if validation:
            validation_set = DatasetVideo2VideoAndAction(path=valsets_path, x_pattern=[0,1], y_pattern=[1],
                                                    demo_percentage=.9, cameras=cameras, validation=True)
    else:
        if train:
            train_set = DatasetVideo2Action(path=datasets_path, video_length=horizon, 
                                            demo_percentage=demo_percentage, cameras=cameras,
                                            motion=motion, image_plus_motion=image_plus_motion, action_type=action_type,
                                            data_modality=data_modality, compute_stats=compute_stats, coordinate_system=coordinate_system)
            action_mean = train_set.action_mean
            action_std = train_set.action_std
        if validation:
            validation_set = DatasetVideo2Action(path=valsets_path, video_length=horizon, 
                                                demo_percentage=val_demo_percentage, num_demo_train=num_demo_train, cameras=cameras, validation=True, 
                                                motion=motion, image_plus_motion=image_plus_motion, action_type=action_type,
                                                data_modality=data_modality, action_mean=action_mean, action_std=action_std, 
                                                compute_stats=compute_stats, coordinate_system=coordinate_system)

    if train and validation:
        return train_set, validation_set
    elif train and not validation:
        return train_set
    elif validation and not train:
        return validation_set
    
def load_model(architecture,
               horizon=1,
               results_path='',
               latent_dim=0,
               cameras=['frontview_image', 'sideview_image'],
               motion=False,
               image_plus_motion=False,
               num_mlp_layers=3, # to be extracted
               vit_patch_size=0, 
               resnet_layers_num=18, # to be extracted
               idm_model_name='',
               fdm_model_name='',
               freeze_idm=None,
               freeze_fdm=None,
               action_type='pose',
               data_modality='rgb', # to be extracted
):
    if architecture == 'direct_cnn_mlp':
        if data_modality == 'voxel':
            model = PoseExtractionCNN3D(latent_dim=latent_dim, 
                                        motion=motion, 
                                        image_plus_motion=image_plus_motion,
                                        num_mlp_layers=num_mlp_layers)
        else:
            model = ActionExtractionCNN(latent_dim=latent_dim, 
                                        video_length=horizon, 
                                        motion=motion, 
                                        image_plus_motion=image_plus_motion,
                                        num_mlp_layers=num_mlp_layers)
    elif architecture == 'direct_cnn_vit':
        model = ActionExtractionViT(latent_dim=latent_dim, 
                                    video_length=horizon, 
                                    motion=motion, 
                                    image_plus_motion=image_plus_motion,
                                    vit_patch_size=vit_patch_size)
    elif architecture == 'direct_resnet_mlp':
        resnet_version = 'resnet' + str(resnet_layers_num)
        
        if 'action' in action_type:
            num_classes = 7
        elif action_type == 'position' or action_type == 'delta_position':
            num_classes = 3
        elif action_type == 'pose' or action_type == 'delta_pose':
            num_classes = 9
        elif action_type == 'position+gripper':
            num_classes = 5
        elif action_type == 'delta_position+gripper':
            num_classes = 4
        
        if data_modality == 'voxel' or data_modality == 'rgbd' or data_modality == 'cropped_rgbd':
            input_channels = 4 * horizon * len(cameras)
        elif data_modality == 'rgb' or 'color_mask_depth':
            input_channels = 3 * horizon * len(cameras)
        elif data_modality == 'cropped_rgbd+color_mask':
            input_channels = 6 * horizon * len(cameras)
        elif data_modality == 'cropped_rgbd+color_mask_depth':
            input_channels = 7 * horizon * len(cameras)
            
        if data_modality == 'rgb' or data_modality == 'color_mask_depth':
            model = ActionExtractionResNet(resnet_version=resnet_version, video_length=horizon, in_channels=3*len(cameras), action_length=1, num_classes=num_classes, num_mlp_layers=num_mlp_layers)
        elif data_modality == 'rgbd' or data_modality == 'cropped_rgbd':
            model = ActionExtractionResNet(resnet_version=resnet_version, video_length=horizon, in_channels=4*len(cameras), action_length=1, num_classes=num_classes, num_mlp_layers=num_mlp_layers)
        elif data_modality == 'cropped_rgbd+color_mask_depth':
            model = ActionExtractionResNet(resnet_version=resnet_version, video_length=horizon, in_channels=7*len(cameras), action_length=1, num_classes=num_classes, num_mlp_layers=num_mlp_layers)
        elif data_modality == 'cropped_rgbd+color_mask':
            model = ActionExtractionResNet(resnet_version=resnet_version, video_length=horizon, in_channels=6*len(cameras), action_length=1, num_classes=num_classes, num_mlp_layers=num_mlp_layers)
        elif data_modality == 'voxel':
            if resnet_layers_num == 18:
                model = resnet18_3d(input_channels=input_channels, num_classes=num_classes, num_mlp_layers=num_mlp_layers)
            elif resnet_layers_num == 34:
                model = resnet34_3d(input_channels=input_channels, num_classes=num_classes, num_mlp_layers=num_mlp_layers)
            elif resnet_layers_num == 50:
                model = resnet50_3d(input_channels=input_channels, num_classes=num_classes, num_mlp_layers=num_mlp_layers)
            elif resnet_layers_num == 101:
                model = resnet101_3d(input_channels=input_channels, num_classes=num_classes, num_mlp_layers=num_mlp_layers)
            elif resnet_layers_num == 152:
                model = resnet152_3d(input_channels=input_channels, num_classes=num_classes, num_mlp_layers=num_mlp_layers)
            elif resnet_layers_num == 200:
                model = resnet200_3d(input_channels=input_channels, num_classes=num_classes, num_mlp_layers=num_mlp_layers)
                
    # elif architecture == 'flownet2':
    #     if 'action' in action_type:
    #         num_classes = 7
    #     elif action_type == 'position':
    #         num_classes = 3
    #     elif action_type == 'pose':
    #         num_classes = 9
        
    #     if data_modality == 'voxel' or data_modality == 'rgbd':
    #         input_channels = 4 * horizon
    #     elif data_modality == 'rgb':
    #         input_channels = 3 * horizon
            
    #     model = FlowNet2PoseEstimation(video_length=horizon, in_channels=input_channels // horizon, num_classes=num_classes, version=flownet_verison,
    #                                    num_mlp_layers=num_mlp_layers)
        
    elif architecture == 'latent_encoder_cnn_unet':
        model = LatentEncoderPretrainCNNUNet(latent_dim=latent_dim, video_length=horizon) # doesn't support motion
    elif architecture == 'latent_encoder_resnet_unet':
        model = LatentEncoderPretrainResNetUNet(resnet_version='resnet' + str(resnet_layers_num), video_length=horizon)
    elif 'latent_decoder' in architecture:
        idm_model_path = str(Path(results_path)) + f'/{idm_model_name}'
        latent_dim = int(re.search(r'_lat(.*?)_', idm_model_name).group(1))
        if architecture == 'latent_decoder_mlp':
            model = LatentDecoderMLP(idm_model_path, 
                                     latent_dim=latent_dim, 
                                     video_length=horizon, 
                                     latent_length=horizon-1, 
                                     mlp_layers=num_mlp_layers)
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
                                                       mlp_layers=num_mlp_layers)
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
                                                        freeze_fdm=freeze_fdm,
                                                        num_mlp_layers=num_mlp_layers)
        elif architecture == 'latent_decoder_aux_combined_vit':
            fdm_model_path = str(Path(results_path)) + f'/{fdm_model_name}'
            model = LatentDecoderAuxiliaryCombinedViT(idm_model_path, 
                                                        latent_dim=latent_dim, 
                                                        video_length=horizon, 
                                                        freeze_idm=freeze_idm)

    return model


def load_trained_model(model, results_path, trained_model_name, device):
    conv_model_path = f"{results_path}/{trained_model_name}"
    
    mlp_model_name_string = trained_model_name.replace('_resnet-', '_mlp-')
    mlp_model_path = f"{results_path}/{mlp_model_name_string}"
    
    conv_state_dict = torch.load(conv_model_path)
    mlp_state_dict = torch.load(mlp_model_path)

    model.conv.load_state_dict(conv_state_dict)
    model.mlp.load_state_dict(mlp_state_dict)
    
    model.conv.to(device)
    model.mlp.to(device)
    
    return model

import matplotlib.pyplot as plt

def check_dataset(inputs, labels):
    # To be called in trainer on inputs and labels for checking dataset correctness
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    for i, ax in enumerate(axes.flat):
        img = inputs[i].cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())

        ax.imshow(img)
        ax.axis('off')

        ax.set_title(f"{labels[i].cpu().numpy()}", fontsize=8)

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()