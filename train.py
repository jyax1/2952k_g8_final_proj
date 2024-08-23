import argparse
from models.direct_cnn_mlp import ActionExtractionCNN
from models.direct_cnn_vit import ActionExtractionViT
from models.latent_cnn_unet import ActionExtractionCNNUNet
from datasets import DatasetVideo2DeltaAction, DatasetVideo
from trainer import Trainer
from pathlib import Path

oscar = True
if oscar:
    dp = '/users/ysong135/scratch/datasets'
    b = 88
else:
    dp = '/home/yilong/Documents/videopredictor/datasets'
    b = 16

def train(args):

    results_path= str(Path(args.datasets_path).parent) + '/ae_results/'
    model_name = f'{args.architecture}_lat_{args.latent_dim}_m_{args.motion}_ipm_{args.image_plus_motion}'

    # Instantiate model
    if args.architecture == 'direct_cnn_mlp':
        model = ActionExtractionCNN(latent_dim=args.latent_dim, video_length=args.horizon, 
                                    motion=args.motion, image_plus_motion=args.image_plus_motion)
    elif args.architecture == 'direct_cnn_vit':
        model = ActionExtractionViT(latent_dim=args.latent_dim, video_length=args.horizon, 
                                    motion=args.motion, image_plus_motion=args.image_plus_motion)
    elif args.architecture == 'latent_cnn_unet':
        model = ActionExtractionCNNUNet(latent_dim=args.latent_dim, video_length=args.horizon) # doesn't support motion

    # Instandiate datasets
    if 'latent' in args.architecture:
        train_set = DatasetVideo(path=args.datasets_path, x_pattern=[0,1], y_pattern=[1],
                                            demo_percentage=0.9, cameras=['frontview_image'])
        validation_set = DatasetVideo(path=args.datasets_path, x_pattern=[0,1], y_pattern=[1],
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

    parser.add_argument('--architecture', '-a', type=str, default='direct_cnn_mlp', 
                        choices=['direct_cnn_mlp', 'direct_cnn_vit', 'latent_cnn_unet'], help='Model architecture to train')
    parser.add_argument('--datasets_path', '-dp', type=str, default=dp, help='Path to the datasets')
    parser.add_argument('--latent_dim', '-ld', type=int, default=32, help='Latent dimension (sqrt of size)')
    parser.add_argument('--epoch', '-e', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--batch_size', '-b', type=int, default=b, help='Batch size')
    parser.add_argument('--motion', '-m', action='store_true', help='Train only with motion')
    parser.add_argument('--image_plus_motion', '-ipm', action='store_true', help='Add motion preprocess to training data')
    parser.add_argument('--horizon', '-hr', type=int, default=2, help='Length of the video')

    args = parser.parse_args()
    assert 128 % args.latent_dim == 0, "latent_dim must divide 128 evenly."
    assert args.horizon > 1, "Video length must be greater or equal to 2"

    train(args)
