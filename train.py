import argparse
from models.direct_cnn import ActionExtractionCNN
from datasets import DatasetVideo2Action
from trainer import Trainer
from pathlib import Path

def train(args):

    results_path= str(Path(args.datasets_path).parent) + '/results/'
    model_name = f'{args.architecture}_lat_{args.latent_size}'
    
    # Instantiate model
    if args == 'direct_unet':
        model = ActionExtractionCNN(latent_size=args.latent_dim)

    # Instandiate datasets
    train_set = DatasetVideo2Action(path=args.datasets_path, video_length=2, demo_percentage=0.9, cameras=['frontview_image'])
    validation_set = DatasetVideo2Action(path=args.datasets_path, video_length=2, demo_percentage=0.9, caemras=['frontview_image'], validation=True)

    # Instantiate the trainer
    trainer = Trainer(model, train_set, validation_set, results_path=results_path, model_name=model_name, batch_size=args.batch_size, epochs=args.epoch)

    # Train the model
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train action extraction model")

    parser.add_argument('--architecture', '-a', type=str, default='direct_unet', choices=['direct_unet'], help='Model architecture to train')
    parser.add_argument('--datasets_path', '-dp', type=str, default='/home/yilong/Documents/videopredictor/datasets', help='Path to the datasets')
    parser.add_argument('--latent_size', '-ls', type=int, default=16, help='Latent size')
    parser.add_argument('--epoch', '-e', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--batch_size', '-b', type=int, default=2, help='Batch size')

    args = parser.parse_args()
    assert 128 % args.latent_dim == 0, "latent_dim must divide 128 evenly."

    train(args)