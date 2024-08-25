import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from accelerate import Accelerator
from models.direct_cnn_mlp import ActionExtractionCNN
from models.direct_cnn_vit import ActionExtractionViT
from models.latent_cnn_unet import ActionExtractionCNNUNet
from models.latent_decoders import LatentDecoderMLP, LatentDecoderTransformer, LatentDecoderObsConditionedUNetMLP
import csv
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_set, validation_set, results_path, model_name, batch_size=32, epochs=100, lr=0.001):
        self.accelerator = Accelerator()
        self.model = model
        self.model_name = model_name
        self.train_set = train_set
        self.validation_set = validation_set
        self.results_path = results_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.val_losses = []

        self.device = self.accelerator.device
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        self.validation_loader = DataLoader(validation_set, batch_size=self.batch_size, shuffle=False)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Prepare the model, optimizer, and dataloaders for distributed training
        self.model, self.optimizer, self.train_loader, self.validation_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.validation_loader
        )

        if not os.path.exists(results_path):
            os.makedirs(results_path)

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            epoch_progress = tqdm(total=len(self.train_loader), desc=f"Epoch [{epoch + 1}/{self.epochs}]", position=0, leave=True)

            for i, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                self.accelerator.backward(loss)
                self.optimizer.step()

                running_loss += loss.item()

                if i % 5 == 4:  # Update every 5 batches
                    avg_loss = running_loss / 10
                    epoch_progress.set_postfix({'Loss': f'{avg_loss:.4f}'})
                    running_loss = 0.0

                # Update tqdm progress bar
                epoch_progress.update(1)

            # Close the progress bar after the epoch ends
            epoch_progress.close()

            # Validate
            val_loss = self.validate()
            print(f'Epoch [{epoch + 1}/{self.epochs}] ended, Validation Loss: {val_loss:.4f}')
            self.val_losses.append(val_loss)

            if epoch % 10 == 9:
                self.save_model(epoch + 1)
                self.val_losses = []

    def validate(self):
        self.model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in self.validation_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_val_loss += loss.item()
        return total_val_loss / len(self.validation_loader)

    def save_model(self, epoch):
        if isinstance(self.model, ActionExtractionCNN):
            torch.save(self.model.frames_convolution_model.state_dict(), os.path.join(self.results_path, f'f_conv_{self.model_name}-{epoch}.pth'))
            torch.save(self.model.action_mlp_model.state_dict(), os.path.join(self.results_path, f'a_mlp_{self.model_name}-{epoch}.pth'))
            with open(os.path.join(self.results_path, f'd_cnn_mlp_{self.model_name}_val.csv'), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for value in self.val_losses:
                    writer.writerow([value])

        if isinstance(self.model, ActionExtractionViT):
            torch.save(self.model.frames_convolution_model.state_dict(), os.path.join(self.results_path, f'f_conv_{self.model_name}-{epoch}.pth'))
            torch.save(self.model.action_transformer_model.state_dict(), os.path.join(self.results_path, f'a_vit_{self.model_name}-{epoch}.pth'))
            with open(os.path.join(self.results_path, f'd_cnn_vit_{self.model_name}_val.csv'), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for value in self.val_losses:
                    writer.writerow([value])
        
        if isinstance(self.model, ActionExtractionCNNUNet):
            torch.save(self.model.idm.state_dict(), os.path.join(self.results_path, f'idm_{self.model_name}-{epoch}.pth'))
            torch.save(self.model.fdm.state_dict(), os.path.join(self.results_path, f'fdm_{self.model_name}-{epoch}.pth'))
            with open(os.path.join(self.results_path, f'lat_{self.model_name}_val.csv'), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for value in self.val_losses:
                    writer.writerow([value])

        if isinstance(self.model, LatentDecoderMLP):
            torch.save(self.model.mlp.state_dict(), os.path.join(self.results_path, f'{self.model_name}-{epoch}.pth'))
            with open(os.path.join(self.results_path, f'{self.model_name}_val.csv'), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for value in self.val_losses:
                    writer.writerow([value])

        if isinstance(self.model, LatentDecoderTransformer):
            torch.save(self.model.transformer.state_dict(), os.path.join(self.results_path, f'{self.model_name}-{epoch}.pth'))
            with open(os.path.join(self.results_path, f'{self.model_name}_val.csv'), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for value in self.val_losses:
                    writer.writerow([value])

        if isinstance(self.model, LatentDecoderObsConditionedUNetMLP):
            torch.save(self.model.unet.state_dict(), os.path.join(self.results_path, f'{self.model_name}_unet-{epoch}.pth'))
            torch.save(self.model.mlp.state_dict(), os.path.join(self.results_path, f'{self.model_name}_mlp-{epoch}.pth'))
            with open(os.path.join(self.results_path, f'{self.model_name}_val.csv'), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for value in self.val_losses:
                    writer.writerow([value])
        else:
            print('Model not supported')