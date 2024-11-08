# spcify model
# load model
import argparse
from utils.utils import *
from validation_visualization import *

class PoseEstimator():
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
    
    def forward_conv(self, x):
        return self.encoder(x)
    
    def forward_mlp(self, x):
        return self.decoder(x)
    
    def forward(self, x):
        return self.decoder(self.encoder(x))
    
def validate_pose_estimator(args):
    architecture = 'direct_resnet_mlp'
    match = re.search(r'res(\d+)', args.trained_model_name)
    
    if "_rgb_" in args.trained_model_name:
        data_modality = 'rgb'
    else:
        data_modality = 'voxel'
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 100

    if match:
        resnet_layers_num = int(match.group(1))
    else:
        resnet_layers_num = 18

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
        action_type='pose',
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