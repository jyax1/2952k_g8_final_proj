import argparse
from models.direct_cnn_mlp import ActionExtractionCNN
from models.direct_cnn_vit import ActionExtractionViT
from models.latent_cnn_unet import ActionExtractionCNNUNet
from models.latent_decoders import LatentDecoderMLP, LatentDecoderTransformer, LatentDecoderObsConditionedUNetMLP, LatentDecoderAuxiliarySeparateUNetMLP, LatentDecoderAuxiliarySeparateUNetTransformer, LatentDecoderAuxiliaryCombinedViT
from datasets import DatasetVideo2DeltaAction, DatasetVideo, DatasetVideo2VideoAndAction
from trainer import Trainer
from pathlib import Path
import re

'''
Temporary
'''
oscar = False
if oscar:
    dp = '/users/ysong135/scratch/datasets_debug'
    b = 88
else:
    dp = '/home/yilong/Documents/datasets'
    b = 16
'''
Temporary
'''

def train(args):

    results_path= str(Path(args.datasets_path).parent) + '/ae_results/'
    model_name = f'{args.architecture}_lat_{args.latent_dim}_m_{args.motion}_ipm_{args.image_plus_motion}'

    # Instantiate model
    if args.architecture == 'direct_cnn_mlp':
        model = ActionExtractionCNN(latent_dim=args.latent_dim, 
                                    video_length=args.horizon, 
                                    motion=args.motion, 
                                    image_plus_motion=args.image_plus_motion)
    elif args.architecture == 'direct_cnn_vit':
        model = ActionExtractionViT(latent_dim=args.latent_dim, 
                                    video_length=args.horizon, 
                                    motion=args.motion, 
                                    image_plus_motion=args.image_plus_motion)
    elif args.architecture == 'latent_cnn_unet':
        model = ActionExtractionCNNUNet(latent_dim=args.latent_dim, video_length=args.horizon) # doesn't support motion
    elif 'latent_decoder' in args.architecture:
        idm_model_path = str(Path(results_path)) + f'/{args.idm_model_name}'
        latent_dim = int(re.search(r'lat_(.*?)_', args.idm_model_name).group(1))
        if args.architecture == 'latent_decoder_mlp':
            model = LatentDecoderMLP(idm_model_path, 
                                     latent_dim=latent_dim, 
                                     video_length=args.horizon, 
                                     latent_length=args.horizon-1, 
                                     mlp_layers=10)
        elif args.architecture == 'latent_decoder_vit':
            model = LatentDecoderTransformer(idm_model_path, 
                                             latent_dim=latent_dim, 
                                             video_length=args.horizon, 
                                             latent_length=args.horizon-1)
        elif args.architecture == 'latent_decoder_obs_conditioned_unet_mlp':
            model = LatentDecoderObsConditionedUNetMLP(idm_model_path, 
                                                       latent_dim=latent_dim, 
                                                       video_length=args.horizon, 
                                                       latent_length=args.horizon-1, 
                                                       mlp_layers=10)
        elif args.architecture == 'latent_decoder_aux_separate_unet_vit':
            fdm_model_path = str(Path(results_path)) + f'/{args.fdm_model_name}'
            model_name = model_name + '_fidm' if args.freeze_idm else model_name
            model_name = model_name + '_ffdm' if args.freeze_fdm else model_name
            model = LatentDecoderAuxiliarySeparateUNetTransformer(idm_model_path, 
                                                        fdm_model_path, 
                                                        latent_dim=latent_dim, 
                                                        video_length=args.horizon, 
                                                        freeze_idm=args.freeze_idm, 
                                                        freeze_fdm=args.freeze_fdm)
        elif args.architecture == 'latent_decoder_aux_separate_unet_mlp':
            fdm_model_path = str(Path(results_path)) + f'/{args.fdm_model_name}'
            model_name = model_name + '_fidm' if args.freeze_idm else model_name
            model_name = model_name + '_ffdm' if args.freeze_fdm else model_name
            model = LatentDecoderAuxiliarySeparateUNetMLP(idm_model_path, 
                                                        fdm_model_path, 
                                                        latent_dim=latent_dim, 
                                                        video_length=args.horizon, 
                                                        freeze_idm=args.freeze_idm, 
                                                        freeze_fdm=args.freeze_fdm)
        elif args.architecture == 'latent_decoder_aux_combined_vit':
            fdm_model_path = str(Path(results_path)) + f'/{args.fdm_model_name}'
            model_name = model_name + '_fidm' if args.freeze_idm else model_name
            model_name = model_name + '_ffdm' if args.freeze_fdm else model_name
            model = LatentDecoderAuxiliaryCombinedViT(idm_model_path, 
                                                        latent_dim=latent_dim, 
                                                        video_length=args.horizon, 
                                                        freeze_idm=args.freeze_idm)

        model_name = f'{args.architecture}_lat_{latent_dim}_m_{args.motion}_ipm_{args.image_plus_motion}'

    # Instandiate datasets
    if 'latent' in args.architecture and 'decoder' not in args.architecture and 'aux' not in args.architecture:
        train_set = DatasetVideo(path=args.datasets_path, x_pattern=[0,1], y_pattern=[1],
                                            demo_percentage=0.9, cameras=['frontview_image'])
        validation_set = DatasetVideo(path=args.datasets_path, x_pattern=[0,1], y_pattern=[1],
                                                    demo_percentage=0.9, cameras=['frontview_image'], validation=True)
    elif 'latent' in args.architecture and 'aux' in args.architecture:
        train_set = DatasetVideo2VideoAndAction(path=args.datasets_path, x_pattern=[0,1], y_pattern=[1],
                                            demo_percentage=0.9, cameras=['frontview_image'])
        validation_set = DatasetVideo2VideoAndAction(path=args.datasets_path, x_pattern=[0,1], y_pattern=[1],
                                                    demo_percentage=0.9, cameras=['frontview_image'], validation=True)
    else:
        train_set = DatasetVideo2DeltaAction(path=args.datasets_path, video_length=args.horizon, 
                                            demo_percentage=0.9, cameras=['frontview_image'],
                                            motion=args.motion, image_plus_motion=args.image_plus_motion)
        validation_set = DatasetVideo2DeltaAction(path=args.datasets_path, video_length=args.horizon, 
                                                demo_percentage=0.9, cameras=['frontview_image'], validation=True, 
                                                motion=args.motion, image_plus_motion=args.image_plus_motion)

    # Instantiate the trainer
    trainer = Trainer(model, train_set, validation_set, results_path=results_path, model_name=model_name, batch_size=args.batch_size, epochs=args.epoch)

    # Train the model
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train action extraction model")

    parser.add_argument(
        '--architecture', '-a', 
        type=str, 
        default='direct_cnn_mlp', 
        choices=['direct_cnn_mlp', 
                    'direct_cnn_vit', 
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
    parser.add_argument(
        '--latent_dim', '-ld', 
        type=int, 
        default=32, 
        help='Latent dimension (sqrt of size)'
    )
    parser.add_argument(
        '--epoch', '-e', 
        type=int, 
        default=100, 
        help='Number of epochs to train'
    )
    parser.add_argument(
        '--batch_size', '-b', 
        type=int, 
        default=b, 
        help='Batch size'
    )
    parser.add_argument(
        '--motion', 
        '-m', 
        action='store_true', 
        help='Train only with motion'
    )
    parser.add_argument(
        '--image_plus_motion', '-ipm', 
        action='store_true', 
        help='Add motion preprocess to training data'
    )
    parser.add_argument(
        '--horizon', '-hr', 
        type=int, 
        default=2, 
        help='Length of the video'
    )
    parser.add_argument(
        '--idm_model_name', '-idm', 
        type=str, 
        default='', 
        help='Path to pretrained latent IDM model'
    )
    parser.add_argument(
        '--fdm_model_name', '-fdm',
        type=str,
        default='',
        help='Path to pretrained latent FDM model'
    )
    parser.add_argument(
        '--freeze_idm', '-fidm',
        action='store_true',
        help='Freeze IDM model for auxiliary training'
    )
    parser.add_argument(
        '--freeze_fdm', '-ffdm',
        action='store_true',
        help='Freeze FDM model for auxiliary training'
    )

    args = parser.parse_args()
    assert 128 % args.latent_dim == 0, "latent_dim must divide 128 evenly."
    assert args.horizon > 1, "Video length must be greater or equal to 2"

    if args.freeze_idm or args.freeze_fdm:
        assert 'aux' in args.architecture and 'latent_decoder' in args.architecture

    if 'latent_decoder' in args.architecture:
        assert args.idm_model_name != ''

        if 'aux' in args.architecture:
            assert args.fdm_model_name != ''

    train(args)
