import argparse
from utils.utils import *
from config import ARCHITECTURES
import torch

'''
Temporary
'''
oscar = True
if oscar:
    dp = '/users/ysong135/scratch/datasets_debug'
else:
    dp = '/home/yilong/Documents/datasets'
'''
Temporary
'''
    
def visualize(model, validation_set, device='cuda'):
    model.eval()
    
    # Move the model to the appropriate device (GPU in this case)
    model.to(device)
    
    # Initialize lists to store feature maps and predictions
    feature_maps = []
    predictions = []
    
    # Get the first 20 images and corresponding labels (if available) from the validation set
    images = [validation_set[i][0] for i in range(20)]
    labels = [validation_set[i][1] for i in range(20)] if isinstance(validation_set[0], tuple) else None
    
    # Stack images into a batch (first axis is batch size)
    images_batch = torch.stack(images)
    
    # Move images to the GPU
    images_batch = images_batch.to(device)
    
    # Run the images through the model on the GPU
    with torch.no_grad():  # No need to calculate gradients
        feature_map = model.idm(images_batch)  # Get feature map from IDM
        past_observations = images_batch[:, :-3, :, :]  # Exclude the last image from the sequence
        prediction = model.fdm(feature_map, past_observations)  # Get prediction from FDM

        # Store feature maps and predictions for each sample
        feature_maps.append(feature_map.cpu())
        predictions.append(prediction.cpu())

def validate(architecture, model_paths=[]):
    validation_datasets = load_datasets(architecture)
    return None

def load_params_from_model_name(model_name):
    '''
    exampe model_name: 
    '''
    params = {}

    # set params['architecture']
    for architecture in ARCHITECTURES:
        if architecture in model_name:
            params['architecture'] = architecture
            break
    
    # set params['latent_dim']
    valid_latent_dims = [4, 8, 16, 32]
    match = re.search(r'_lat\D*(\d+)', model_name)
    if match:
        latent_dim = int(match.group(1))
        if latent_dim in valid_latent_dims:
            params['latent_dim'] = latent_dim

    # set params['motion']
    match = re.search(r'_m\D*(True|False)', model_name)
    if match:
        motion = match.group(1) == "True"
        params['motion'] = motion
            
    # set params['image_plus_motion']
    match = re.search(r'_ipm\D*(True|False)', model_name)
    if match:
        image_plus_motion = match.group(1) == "True"
        params['image_plus_motion'] = image_plus_motion

    # set params['vit_patch_size']
    match = re.search(r'_vps(\d+)', model_name)
    if match:
        vit_patch_size = int(match.group(1))
        params['vit_patch_size'] = vit_patch_size
    else:
        params['vit_patch_size'] = 4

    # set params['resnet_layers_num']
    match = re.search(r'_res(\d+)', model_name)
    if match:
        resnet_layers_num = int(match.group(1))
        params['resnet_layers_num'] = resnet_layers_num
    else:
        params['resnet_layers_num'] = 0

    return params


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Validation or visualization of trained models")

    parser.add_argument(
        '--model_name', '-mn', 
        type=str,
        help='Pretrained model to validate'
    )

    parser.add_argument(
        '--datasets_path', '-dp', 
        type=str, 
        default=dp, 
        help='Path to the datasets'
    )

    args = parser.parse_args()

    results_path= str(Path(args.datasets_path).parent) + '/ae_results/'

    params = load_params_from_model_name(args.model_name)

    idm_model_name = ''
    fdm_model_name = ''
    if 'latent' in params['architecture']:
        if 'idm' in args.model_name:
            idm_model_name = args.model_name
            fdm_model_name = args.model_name.replace('idm', 'fdm')
        elif 'fdm' in args.model_name:
            fdm_model_name = args.model_name
            idm_model_name = args.model_name.replace('fdm', 'idm')
        idm_model_name = str(Path(results_path)) + f'/{idm_model_name}'
        fdm_model_name = str(Path(results_path)) + f'/{fdm_model_name}'

    validation_set = load_datasets(
        params['architecture'],
        args.datasets_path,
        train=False,
        validation=True,
        horizon=2,
        demo_percentage=.9,
        cameras=['frontview_image'],
        motion=params['motion'],
        image_plus_motion=params['image_plus_motion']
        )

    model = load_model(
        params['architecture'],
        horizon=2,
        results_path=results_path,
        latent_dim=params['latent_dim'],
        motion=params['motion'],
        image_plus_motion=params['image_plus_motion'],
        vit_patch_size=params['vit_patch_size'],
        resnet_layers_num=params['resnet_layers_num'],
        idm_model_name=idm_model_name,
        fdm_model_name=fdm_model_name,
        freeze_idm=True,
        freeze_fdm=True ## these two parameters don't matter here
        )

    if 'latent' in params['architecture'] and 'decoder' not in params['architecture']:
        model.idm.load_state_dict(torch.load(idm_model_name))
        model.fdm.load_state_dict(torch.load(fdm_model_name))
        visualize(model, validation_set)
    else:
        validate(args.dataset_path, args.architecture)