import argparse
from models.direct_cnn import ActionExtractionCNN
from datasets import DatasetVideo2Action
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
    
    args.latent_size = args.latent_size**2 #

    results_path= str(Path(args.datasets_path).parent) + '/ae_results/'
    model_name = f'{args.architecture}_lat_{args.latent_size}'
    
    # Instantiate model
    if args.architecture == 'direct_unet':
        model = ActionExtractionCNN(latent_size=args.latent_size)

    # Instandiate datasets
    train_set = DatasetVideo2Action(path=args.datasets_path, video_length=2, demo_percentage=0.9, cameras=['frontview_image'])
    validation_set = DatasetVideo2Action(path=args.datasets_path, video_length=2, demo_percentage=0.9, cameras=['frontview_image'], validation=True)

    # Instantiate the trainer
    trainer = Trainer(model, train_set, validation_set, results_path=results_path, model_name=model_name, batch_size=args.batch_size, epochs=args.epoch)

    # Train the model
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train action extraction model")

    parser.add_argument('--architecture', '-a', type=str, default='direct_unet', choices=['direct_unet'], help='Model architecture to train')
    parser.add_argument('--datasets_path', '-dp', type=str, default=dp, help='Path to the datasets')
    parser.add_argument('--latent_size', '-ls', type=int, default=32, help='Latent size')
    parser.add_argument('--epoch', '-e', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--batch_size', '-b', type=int, default=b, help='Batch size')

    args = parser.parse_args()
    assert 128 % args.latent_size == 0, "latent_size must divide 128 evenly."

    train(args)
