import argparse
from utils.utils import *
from trainer import Trainer
from config import *

'''
Temporary
'''
oscar = True
if oscar:
    dp = '/users/ysong135/scratch/datasets_debug'
    b = 88
    rp = '/users/ysong135/Documents/action_extractor/results'
else:
    dp = '/home/yilong/Documents/datasets'
    b = 16
    rp = '/home/yilong/Documents/action_extractor/results'
'''
Temporary
'''

def train(args):

    model_name = f'''{args.architecture}_cam{args.cameras}_emb{args.embodiments}_lat{args.latent_dim}_res{args.resnet_layers_num}_vps{args.vit_patch_size}_fidm{args.freeze_idm}_ffdm{args.freeze_fdm}_{args.note}'''

    # Instantiate model
    model = load_model(
        args.architecture,
        horizon=args.horizon,
        results_path=args.results_path,
        latent_dim=args.latent_dim,
        motion=args.motion,
        image_plus_motion=args.image_plus_motion,
        vit_patch_size=args.vit_patch_size,
        resnet_layers_num=args.resnet_layers_num,
        idm_model_name=args.idm_model_name,
        fdm_model_name=args.fdm_model_name,
        freeze_idm=args.freeze_idm,
        freeze_fdm=args.freeze_fdm,
        dinov2=args.dinov2
        )

    # Instandiate datasets
    train_set, validation_set = load_datasets(
        args.architecture, 
        args.datasets_path, 
        train=True,
        validation=True,
        horizon=args.horizon,
        demo_percentage=args.demo_percentage,
        cameras=args.cameras,
        motion=args.motion,
        image_plus_motion=args.image_plus_motion
        )

    # Instantiate the trainer
    trainer = Trainer(
        model, 
        train_set, 
        validation_set, 
        results_path=args.results_path, 
        model_name=model_name, 
        batch_size=args.batch_size, 
        epochs=args.epoch
        )

    # Train the model
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train action extraction model")

    parser.add_argument(
        '--architecture', '-a', 
        type=str, 
        default='direct_cnn_mlp', 
        choices=ARCHITECTURES,
        help='Model architecture to train'
    )
    parser.add_argument(
        '--datasets_path', '-dp', 
        type=str, 
        default=dp, 
        help='Path to the datasets'
    )
    parser.add_argument(
        '--results_path', '-rp', 
        type=str, 
        default=rp, 
        help='Path to where the results should be stored'
    )
    parser.add_argument(
        '--latent_dim', '-ld', 
        type=int, 
        default=32,
        choices=VALID_LATENT_DIMS,
        help='Latent dimension (sqrt of size)'
    )
    parser.add_argument(
        '--epoch', '-e', 
        type=int, 
        default=1, 
        help='Number of epochs to train'
    )
    parser.add_argument(
        '--batch_size', '-b', 
        type=int, 
        default=b, 
        help='Batch size'
    )
    parser.add_argument(
        '--motion', '-m', 
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
    parser.add_argument(
        '--demo_percentage', '-dpc',
        type=float,
        default=0.9,
        help='Percentage of demos (spread evenly across each task) to use for training'
    )
    parser.add_argument(
        '--vit_patch_size', '-vps',
        type=int,
        default=16,
        help='Patch size to use for the ViT if architecture involves a ViT component'
    )
    parser.add_argument(
        '--resnet_layers_num', '-rln',
        type=int,
        default=0,
        choices=[0, 18, 50],
        help='Number of layers if direct_resnet_mlp architecture is chosen'
    )
    parser.add_argument(
        '--note',
        type=str,
        default='',
        help='Custom note added to the end of the model name'
    )
    parser.add_argument(
        '--cameras', '-c',
        type=str,
        default='frontview_image',
        help='Comma separated list of camera angles to be used for training'
    )
    parser.add_argument(
        '--embodiments', '-emb',
        type=str,
        default='',
        help='Comma separated list of embodiments to be used for training'
    )
    parser.add_argument(
        '--dinov2',
        type=str,
        choices=['dinov2_vits14, dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14', 
                 'dinov2_vits14_reg', 'dinov2_vitb14_reg', 'dinov2_vitl14_reg', 'dinov2_vitg14_reg'],
        default='',
        help='dinov2 model to use for preprocessing'
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
    
    if 'resnet' in args.architecture:
        assert args.resnet_layers_num == 18 or args.resnet_layers_num == 50, "Choose either ResNet-18 or ResNet-50"

    args.cameras = args.cameras.split(',')
    args.embodiments = args.embodiments.split(',')

    print('Arguments:', args) # Check argument correctness in jobs
    train(args)
