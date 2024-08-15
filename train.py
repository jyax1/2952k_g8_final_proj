import argparse
from models.direct_cnn_mlp import ActionExtractionCNN
from models.direct_cnn_vit import ActionExtractionViT
from datasets import DatasetVideo2DeltaAction
from trainer import Trainer
from pathlib import Path

oscar = False
if oscar:
    dp = '/users/ysong135/scratch/datasets'
    b = 88
else:
    dp = '/home/yilong/Documents/videopredictor/datasets'
    b = 16

def train(args):
    
    args.latent_size = args.latent_size**2

    results_path= str(Path(args.datasets_path).parent) + '/ae_results/'
    model_name = f'{args.architecture}_lat_{args.latent_size}_m_{args.motion}_ipm_{args.image_plus_motion}'

    # Instantiate model
    if args.architecture == 'direct_cnn_mlp':
        model = ActionExtractionCNN(latent_size=args.latent_size, video_length=args.horizon, 
                                    motion=args.motion, image_plus_motion=args.image_plus_motion)
    if args.architecture == 'direct_cnn_vit':
        model = ActionExtractionViT(latent_size=args.latent_size, video_length=args.horizon, 
                                    motion=args.motion, image_plus_motion=args.image_plus_motion)

    # Instandiate datasets
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

    parser.add_argument('--architecture', '-a', type=str, default='direct_cnn_mlp', choices=['direct_cnn_mlp', 'direct_cnn_vit'], help='Model architecture to train')
    parser.add_argument('--datasets_path', '-dp', type=str, default=dp, help='Path to the datasets')
    parser.add_argument('--latent_size', '-ls', type=int, default=32, help='Latent size')
    parser.add_argument('--epoch', '-e', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--batch_size', '-b', type=int, default=b, help='Batch size')
    parser.add_argument('--motion', '-m', action='store_true', help='Train only with motion')
    parser.add_argument('--image_plus_motion', '-ipm', action='store_true', help='Add motion preprocess to training data')
    parser.add_argument('--horizon', '-hr', type=int, default=2, help='Length of the video')

    args = parser.parse_args()
    assert 128 % args.latent_size == 0, "latent_size must divide 128 evenly."
    assert args.horizon > 1, "Video length must be greater or equal to 2"

    train(args)
