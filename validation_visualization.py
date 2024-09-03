import argparse
from utils.utils import *
from config import ARCHITECTURES
import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as T

random.seed(0)

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

def visualize_and_save_images(
        normalized_feature_map, 
        scaled_first_images, 
        scaled_prediction, 
        scaled_labels, 
        output_dir='/users/ysong135/Desktop/latent_action_visualization'
):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Number of samples
    num_samples = normalized_feature_map.shape[0]
    
    # Loop through each sample
    for i in range(num_samples):
        # Extract the nth sample from each tensor
        first_image = scaled_first_images[i]
        feature_map = normalized_feature_map[i]
        prediction = scaled_prediction[i]
        true_label = scaled_labels[i]

        # Resize feature_map to (1, 128, 128) and convert to 3 channels
        c = feature_map.shape[2]
        upsampled_feature_map = T.Resize((128, 128))(feature_map)  # Upsample to (1, 128, 128)
        upsampled_feature_map = upsampled_feature_map.repeat(3, 1, 1)  # Convert to (3, 128, 128) by repeating the channel
        
        # Move to CPU and convert upsampled_feature_map to a PIL Image and apply color mapping
        upsampled_feature_map_cpu = upsampled_feature_map.cpu().numpy().transpose(1, 2, 0)  # Transpose to (128, 128, 3)
        colormap = plt.cm.ScalarMappable(cmap='cividis')
        upsampled_feature_map_img = Image.fromarray((colormap.to_rgba(upsampled_feature_map_cpu)[:, :, :3] * 255).astype('uint8'))

        # Convert tensors to PIL Images for concatenation
        first_image_pil = T.ToPILImage()(first_image.cpu())
        prediction_pil = T.ToPILImage()(prediction.cpu())
        true_label_pil = T.ToPILImage()(true_label.cpu())

        # Concatenate the images horizontally
        combined_image = Image.new('RGB', (128 * 4, 128))
        combined_image.paste(first_image_pil, (0, 0))
        combined_image.paste(upsampled_feature_map_img, (128, 0))
        combined_image.paste(prediction_pil, (128 * 2, 0))
        combined_image.paste(true_label_pil, (128 * 3, 0))
        
        # Annotate the images with labels
        plt.figure(figsize=(12, 4))
        plt.imshow(combined_image)
        plt.axis('off')
        plt.text(64, 140, 'current obs', ha='center', fontsize=12)
        plt.text(192, 140, f'latent action ({c}x{c})', ha='center', fontsize=12)
        plt.text(320, 140, 'predicted next obs', ha='center', fontsize=12)
        plt.text(448, 140, 'true next obs', ha='center', fontsize=12)
        
        # Save the annotated image
        save_path = os.path.join(output_dir, f'sample_{i + 1}.png')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    
def visualize(model, validation_set, device='cuda'):
    model.eval()
    model.to(device)
    
    # 20 random samples from validation set
    indices = random.sample(range(len(validation_set)), 20)
    images = [validation_set[i][0] for i in indices]
    labels = [validation_set[i][1] for i in indices] if isinstance(validation_set[0], tuple) else None
    
    # stack
    images_batch, labels_batch = torch.stack(images), torch.stack(labels)
    
    images_batch = images_batch.to(device)
    
    with torch.no_grad():
        # obtain and normalize feature map
        feature_map = model.idm(images_batch) 
        feature_map_min = feature_map.min()
        feature_map_max = feature_map.max()
        normalized_feature_map = (feature_map - feature_map_min) / (feature_map_max - feature_map_min)

        # obtain scale and normalize prediction
        past_observations = images_batch[:, :-3, :, :]
        prediction = model.fdm(feature_map, past_observations)

        scaled_prediction = torch.clamp((prediction * 255).to(torch.uint8), min=0, max=255)

    # scale labels
    scaled_labels = (labels_batch * 255).to(torch.uint8)

    scaled_first_images = (images_batch[:, :3, :, :] * 255).to(torch.uint8)

    visualize_and_save_images(normalized_feature_map, scaled_first_images, scaled_prediction, scaled_labels)

def validate(architecture, model_paths=[]):
    validation_datasets = load_datasets(architecture)
    return None

def load_params_from_model_name(model_name):

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