import argparse
import numpy as np
import torch
import torch.nn as nn
from action_extractor.architectures import ActionExtractionResNet
from action_extractor.architectures import ActionExtractionVariationalResNet
from action_extractor.utils.utils import load_model, load_trained_model, load_datasets

class VariationalEncoder(nn.Module):
    def __init__(self, conv, fc_mu, fc_logvar):
        super(VariationalEncoder, self).__init__()
        self.conv = conv
        self.flatten = nn.Flatten()
        self.fc_mu = fc_mu
        self.fc_logvar = fc_logvar

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class ActionIdentifier(nn.Module):
    def __init__(self, encoder, decoder, stats_path='/home/yilong/Documents/ae_data/random_processing/iiwa16168/action_statistics_delta_position+gripper.npz', 
                 coordinate_system='global', camera_name='frontview'):
        super(ActionIdentifier, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        # Load standardization stats
        stats = np.load(stats_path)
        self.action_mean = torch.tensor(stats['action_mean'], dtype=torch.float32)
        self.action_std = torch.tensor(stats['action_std'], dtype=torch.float32)
        
        # Set coordinate system and camera parameters  
        self.coordinate_system = coordinate_system
        self.camera_name = camera_name
        
        # Select appropriate camera matrix based on camera_name
        from action_extractor.utils.dataset_utils import frontview_R, sideview_R, agentview_R, sideagentview_R
        if camera_name == 'frontview':
            self.R = frontview_R
        elif camera_name == 'sideview':
            self.R = sideview_R  
        elif camera_name == 'agentview':
            self.R = agentview_R
        elif camera_name == 'sideagentview':
            self.R = sideagentview_R
        else:
            raise ValueError(f"Unknown camera name: {camera_name}")
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        action = self.decoder(x)
        
        # Unstandardize
        action = action * self.action_std.to(action.device) + self.action_mean.to(action.device)
        
        # Convert to global coordinates if needed
        if self.coordinate_system in ['camera', 'disentangled']:
            action = self.transform_to_global(action)
        
        return action
    
    def forward(self, x):
        return self.decode(self.encode(x))
    
    def transform_to_global(self, action):
        # Extract position component (first 3 dimensions) and other components
        pos = action[:, :3]
        other = action[:, 3:]
        
        if self.coordinate_system == 'camera':
            # Convert from camera frame to global frame
            pos_homog = torch.cat([pos, torch.ones(pos.shape[0], 1).to(pos.device)], dim=1)
            R_inv = torch.inverse(torch.from_numpy(self.R).float().to(pos.device))
            pos_global = (R_inv @ pos_homog.unsqueeze(-1)).squeeze(-1)[:, :3]
            
        elif self.coordinate_system == 'disentangled':
            # Convert from (x/z, y/z, log(z)) back to (x, y, z)
            x_over_z, y_over_z, log_z = pos[:, 0], pos[:, 1], pos[:, 2]
            z = torch.exp(log_z)
            x = x_over_z * z
            y = y_over_z * z
            pos_global = torch.stack([x, y, z], dim=1)
            
        # Combine transformed position with other components
        return torch.cat([pos_global, other], dim=1)


def load_action_identifier(
    conv_path,
    mlp_path,
    resnet_version,
    video_length,
    in_channels,
    action_length,
    num_classes,
    num_mlp_layers,
    fc_mu_path=None,
    fc_logvar_path=None,
    stats_path='action_statistics_delta_position+gripper.npz',
    coordinate_system='global',
    camera_name='frontview',
    split_layer='avgpool' # The last layer of the encoder
):
    # Build the model
    if fc_mu_path is not None and fc_logvar_path is not None:
        architecture = ActionExtractionVariationalResNet
    else:
        architecture = ActionExtractionResNet
        
    model = architecture(
        resnet_version=resnet_version,
        video_length=video_length,
        in_channels=in_channels,
        action_length=action_length,
        num_classes=num_classes,
        num_mlp_layers=num_mlp_layers
    )

    # Load the saved conv and mlp parts into the model
    if conv_path is not None:
        model.conv.load_state_dict(torch.load(conv_path))
    else:
        model.conv = None

    if mlp_path is not None:
        model.mlp.load_state_dict(torch.load(mlp_path))
    else:
        model.mlp = None
        
    if fc_mu_path is not None:
        model.fc_mu.load_state_dict(torch.load(fc_mu_path))
    else:
        model.fc_mu = None
        
    if fc_logvar_path is not None:
        model.fc_logvar.load_state_dict(torch.load(fc_logvar_path))
    else:
        model.fc_logvar = None

    # Split the conv module into encoder and decoder
    encoder_layers = nn.Sequential()
    decoder_layers = nn.Sequential()
    add_to_decoder = False

    if isinstance(model, ActionExtractionResNet):
        for name, module in model.conv.named_children():
            if not add_to_decoder:
                encoder_layers.add_module(name, module)
                if name == split_layer:
                    add_to_decoder = True  # Start adding to decoder after this layer
            else:
                decoder_layers.add_module(name, module)

        # Combine decoder_layers with model.mlp
        full_decoder = nn.Sequential(decoder_layers, 
            nn.Flatten(), 
            model.mlp
        )
    elif isinstance(model, ActionExtractionVariationalResNet):
        # Initialize variational encoder
        encoder = VariationalEncoder(
            conv=model.conv,
            fc_mu=model.fc_mu,
            fc_logvar=model.fc_logvar
        )

        # Initialize decoder as the MLP part of the model
        decoder = model.mlp

        # Combine decoder_layers as the full_decoder
        full_decoder = nn.Sequential(
            decoder,                # MLP head
            # Add more layers here if necessary
        )
        
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

    # Initialize ActionIdentifier with the new encoder and decoder
    action_identifier = ActionIdentifier(
        encoder=encoder if isinstance(model, ActionExtractionVariationalResNet) else encoder_layers,
        decoder=full_decoder,
        stats_path=stats_path,
        coordinate_system=coordinate_system,
        camera_name=camera_name,
    )

    return action_identifier
    
def validate_pose_estimator(args):
    architecture = 'direct_resnet_mlp'
    data_modality = 'cropped_rgbd+color_mask'
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 100

    resnet_layers_num = 18
    action_type = 'delta_position+gripper'
    video_length = 2

    model = load_model(
        architecture,
        horizon=video_length,
        results_path='',
        latent_dim=0,
        motion=False,
        image_plus_motion=False,
        num_mlp_layers=3, # to be extracted
        vit_patch_size=0, 
        resnet_layers_num=resnet_layers_num, # to be extracted
        idm_model_name='',
        fdm_model_name='',
        freeze_idm=None,
        freeze_fdm=None,
        action_type=action_type,
        data_modality=data_modality # to be extracted
        )
    
    trained_model = load_trained_model(model, args.results_path, args.trained_model_name, device)
    
    validation_set = load_datasets(
        architecture, 
        args.datasets_path, 
        args.datasets_path,
        train=False,
        validation=True,
        horizon=1,
        demo_percentage=0.9,
        cameras=['frontview_image'],
        motion=False,
        image_plus_motion=False,
        action_type='pose',
        data_modality=data_modality
        )
    
    from action_extractor.utility_scripts.validation_visualization import validate_and_record
    validate_and_record(trained_model, validation_set, args.trained_model_name[:-4], batch_size, device)
        
if __name__ == '__main__':
    architecture = ''
    modality = ''
    
    parser = argparse.ArgumentParser(description="Load model for pose estimation")
    parser.add_argument(
        '--trained_model_name', '-mn', 
        type=str, 
        default='direct_resnet_mlp_res18_optadam_lr0.001_mmt0.9_18_coffee_pose_std_rgb_resnet-20-1559.pth', 
        help='trained model to load'
    )
    parser.add_argument(
        '--results_path', '-rp', 
        type=str, 
        default='/home/yilong/Documents/action_extractor/results',
        help='Path to where the results should be stored'
    )
    parser.add_argument(
        '--datasets_path', '-dp', 
        type=str, 
        default='/home/yilong/Documents/ae_data/datasets/mimicgen_core/coffee_rel',
        help='Path to where the datasets are stored'
    )
    
    args = parser.parse_args()
    
    validate_pose_estimator(args)