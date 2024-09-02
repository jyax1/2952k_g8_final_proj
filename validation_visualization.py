import argparse
from utils.utils import load_datasets
from architectures.direct_cnn_mlp import ActionExtractionCNN
from architectures.direct_cnn_vit import ActionExtractionViT
from architectures.latent_cnn_unet import ActionExtractionCNNUNet
from architectures.direct_resnet_mlp import ActionExtractionResNet
from architectures.latent_decoders import LatentDecoderMLP, LatentDecoderTransformer, LatentDecoderObsConditionedUNetMLP, LatentDecoderAuxiliarySeparateUNetMLP, LatentDecoderAuxiliarySeparateUNetTransformer, LatentDecoderAuxiliaryCombinedViT
from datasets import DatasetVideo2DeltaAction, DatasetVideo, DatasetVideo2VideoAndAction
from trainer import Trainer
from pathlib import Path
import re

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
    
def visualize(dataset_path, architecture):
    validation_datasets = load_datasets(architecture)
    return None

def validate(architecture, model_paths=[]):
    validation_datasets = load_datasets(architecture)
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Validation or visualization of trained models")

    parser.add_argument(
        '--architecture', '-a', 
        type=str, 
        default='direct_cnn_mlp', 
        choices=['direct_cnn_mlp', 
                    'direct_cnn_vit',
                    'direct_resnet_mlp', 
                    'latent_cnn_unet', 
                    'latent_decoder_mlp', 
                    'latent_decoder_vit', 
                    'latent_decoder_obs_conditioned_unet_mlp',
                    'latent_decoder_aux_separate_unet_mlp',
                    'latent_decoder_aux_separate_unet_vit',
                    'latent_decoder_aux_combined_unet_mlp',
                    'latent_decoder_aux_combined_vit',
        ],
        help='Model architecture to train'
    )

    parser.add_argument(
        '--datasets_path', '-dp', 
        type=str, 
        default=dp, 
        help='Path to the datasets'
    )

    args = parser.parse_args()

    if 'latent' in args.architecture and 'decoder' not in args.architecture:
        visualize(args.dataset_path, args.architecture)
    else:
        validate(args.dataset_path, args.architecture)