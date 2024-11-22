# spcify model
# load model
import argparse
from utils.utils import *
from validation_visualization import *

class ActionIdentifier():
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
    
    def forward_conv(self, x):
        return self.encoder(x)
    
    def forward_mlp(self, x):
        return self.decoder(x)
    
    def forward(self, x):
        return self.decoder(self.encoder(x))
    
def load_action_identifier(conv_path, mlp_path, resnet_version, video_length, in_channels, action_length, num_classes, num_mlp_layers):
    model = ActionExtractionResNet(
        resnet_version=resnet_version,
        video_length=video_length,
        in_channels=in_channels,
        action_length=action_length,
        num_classes=num_classes,
        num_mlp_layers=num_mlp_layers
    )

    # Load the saved conv and mlp parts into the model
    model.conv.load_state_dict(torch.load(conv_path))
    model.mlp.load_state_dict(torch.load(mlp_path))

    # Initialize ActionIdentifier with the loaded ActionExtractionResNet model
    action_identifier = ActionIdentifier(encoder=model.conv, decoder=model.mlp)
    
    return action_identifier
    
def validate_pose_estimator(args):
    architecture = 'direct_resnet_mlp'
    data_modality = 'cropped_rgbd+color_mask'
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 100

    resnet_layers_num = 18
    action_type = 'delta_position+gripper'

    model = load_model(
        architecture,
        horizon=1,
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