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
oscar = False
if oscar:
    dp = '/users/ysong135/scratch/datasets_debug'
    rp = '/users/ysong135/Documents/action_extractor/results'
else:
    dp = '/home/yilong/Documents/ae_data/datasets'
    rp = '/home/yilong/Documents/action_extractor/results'
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

def validate(model, validation_set, device='cuda'):
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
        actions = model.forward(images_batch)
        
        criterion = nn.MSELoss()
        loss = criterion(actions, labels_batch.to(device))
        
        for i, (label, action) in enumerate(zip(labels_batch, actions)):
            print(f"Row {i + 1}:")
            print(f"Label: {label.tolist()}")
            print(f"Action: {action.tolist()}")
            print()  # Print a blank line for readability

        
        print(f'Validation loss: {loss}')
        
    
def load_params_from_model_name(model_name):
    
    if model_name == '':
        return None

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
    else:
        params['motion'] = False
            
    # set params['image_plus_motion']
    match = re.search(r'_ipm\D*(True|False)', model_name)
    if match:
        image_plus_motion = match.group(1) == "True"
        params['image_plus_motion'] = image_plus_motion
    else:
        params['image_plus_motion'] = False

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
        
    params['resnet_layers_num'] = 50

    return params


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Validation or visualization of trained models")

    parser.add_argument(
        '--encoder_model_name', '-emn', 
        type=str,
        default='',
        help='Pretrained encoder model to validate or visualize'
    )
    
    parser.add_argument(
        '--decoder_model_name', '-dmn', 
        type=str,
        default='',
        help='Pretrained decoder model to validate or visualize'
    )
    
    parser.add_argument(
        '--direct_model_name', '-dirmn',
        type=str,
        default='',
        help='Pretrained direct model to validate or visualize'
    )

    parser.add_argument(
        '--datasets_path', '-dp', 
        type=str, 
        default=dp, 
        help='Path to the datasets'
    )

    args = parser.parse_args()

    results_path= rp

    encoder_params = load_params_from_model_name(args.encoder_model_name)
    decoder_params = load_params_from_model_name(args.decoder_model_name)
    direct_params  = load_params_from_model_name(args.direct_model_name)

    idm_model_name = ''
    fdm_model_name = ''
    if encoder_params != None and 'latent' in encoder_params['architecture']:
        if 'idm' in args.encoder_model_name:
            idm_model_name = args.encoder_model_name
            fdm_model_name = args.encoder_model_name.replace('idm', 'fdm')
        elif 'fdm' in args.encoder_model_name:
            fdm_model_name = args.encoder_model_name
            idm_model_name = args.encoder_model_name.replace('fdm', 'idm')
        idm_model_path = str(Path(results_path)) + f'/{idm_model_name}'
        fdm_model_path = str(Path(results_path)) + f'/{fdm_model_name}'

        
    if direct_params != None:
        if 'resnet' in direct_params['architecture']:
            match = re.search(r'_(mlp|resnet)-\d+-\d+\.pth$', args.direct_model_name)
            if match:
                if match.group(1) == 'mlp':
                    mlp_model_name = args.direct_model_name
                    resnet_model_name = args.direct_model_name.replace('_mlp-', '_resnet-')
                elif match.group(1) == 'resnet':
                    mlp_model_name = args.direct_model_name.replace('_resnet-', '_lmlp-')
                    resnet_model_name = args.direct_model_name
                  
            resnet_model_path = str(Path(results_path)) + f'/{resnet_model_name}'
            mlp_model_path = str(Path(results_path)) + f'/{mlp_model_name}'
        
            
                    
    
    decoder_model_name = args.decoder_model_name
    
    direct_model_name = args.direct_model_name
    
    if decoder_params == None and encoder_params != None:
        architecture = encoder_params['architecture']
    if decoder_params != None:
        architecture = decoder_params['architecture']
    if direct_params != None:
        architecture = direct_params['architecture']

    validation_set = load_datasets(
        architecture,
        args.datasets_path,
        train=False,
        validation=True,
        horizon=2,
        demo_percentage=.9,
        cameras=['frontview_image'],
        motion=False,
        image_plus_motion=False
        )

    if decoder_params == None and encoder_params != None:
        encoder_model = load_model(
            encoder_params['architecture'],
            horizon=2,
            results_path=results_path,
            latent_dim=encoder_params['latent_dim'],
            motion=encoder_params['motion'],
            image_plus_motion=encoder_params['image_plus_motion'],
            num_mlp_layers=10,
            vit_patch_size=encoder_params['vit_patch_size'],
            resnet_layers_num=encoder_params['resnet_layers_num'],
            idm_model_name=idm_model_name,
            fdm_model_name=fdm_model_name,
            freeze_idm=True,
            freeze_fdm=True ## these two parameters don't matter here
            )
    
    if decoder_params != None:
        decoder_model = load_model(
            decoder_params['architecture'],
            horizon=2,
            results_path=results_path,
            latent_dim=decoder_params['latent_dim'],
            motion=decoder_params['motion'],
            image_plus_motion=decoder_params['image_plus_motion'],
            num_mlp_layers=10, #10 for decoder
            vit_patch_size=decoder_params['vit_patch_size'],
            resnet_layers_num=decoder_params['resnet_layers_num'],
            idm_model_name=idm_model_name,
            fdm_model_name=fdm_model_name,
            freeze_idm=True,
            freeze_fdm=True ## these two parameters don't matter here
            )
        
    if direct_params != None:
        direct_model = load_model(
            direct_params['architecture'],
            horizon=2,
            results_path=results_path,
            latent_dim=direct_params['latent_dim'],
            motion=direct_params['motion'],
            image_plus_motion=direct_params['image_plus_motion'],
            num_mlp_layers=3,
            vit_patch_size=direct_params['vit_patch_size'],
            resnet_layers_num=direct_params['resnet_layers_num'],
            idm_model_name=idm_model_name,
            fdm_model_name=fdm_model_name,
            freeze_idm=True,
            freeze_fdm=True ## these two parameters don't matter here
            )
        

    if encoder_params != None and 'latent' in encoder_params['architecture'] and decoder_params == None:
        encoder_model.idm.load_state_dict(torch.load(idm_model_path))
        encoder_model.fdm.load_state_dict(torch.load(fdm_model_path))
        visualize(encoder_model, validation_set)
    elif encoder_params != None and 'latent' in encoder_params['architecture'] and decoder_params != None:
        validate(decoder_model, validation_set)
    elif direct_params != None and 'resnet' in direct_params['architecture']:
        direct_model.resnet.load_state_dict(torch.load(resnet_model_path))
        direct_model.action_mlp.load_state_dict(torch.load(mlp_model_path))
        validate(direct_model, validation_set)