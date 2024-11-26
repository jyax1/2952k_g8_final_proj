import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from architectures.direct_cnn_mlp import ActionExtractionCNN
from architectures.direct_cnn_vit import ActionExtractionViT
from architectures.latent_encoders import LatentEncoderPretrainCNNUNet, LatentEncoderPretrainResNetUNet
from architectures.latent_decoders import *
from architectures.direct_resnet_mlp import ActionExtractionResNet
from architectures.resnet import ResNet3D
import csv
from tqdm import tqdm
from utils.utils import check_dataset

import torch.nn.functional as F

class DeltaControlLoss(nn.Module):
    def __init__(self, direction_weight=1.0):
        super(DeltaControlLoss, self).__init__()
        self.direction_weight = direction_weight

    def forward(self, predictions, targets):
        # Split the vectors
        pred_direction, pred_magnitude = predictions[:, :3], predictions[:, :3].norm(dim=1, keepdim=True)
        target_direction, target_magnitude = targets[:, :3], targets[:, :3].norm(dim=1, keepdim=True)

        # Normalize to get unit vectors for direction
        pred_direction_normalized = F.normalize(pred_direction, dim=1)
        target_direction_normalized = F.normalize(target_direction, dim=1)

        # Compute directional loss (cosine similarity)
        direction_loss = 1 - F.cosine_similarity(pred_direction_normalized, target_direction_normalized).mean()

        # Compute magnitude loss (MSE for magnitude)
        magnitude_loss = F.mse_loss(pred_magnitude, target_magnitude)

        # Combine direction and magnitude loss for the first three components
        vector_loss = (self.direction_weight * direction_loss) + ((1 - self.direction_weight) * magnitude_loss)

        # Compute MSE loss for the last two components
        mse_loss_gripper = F.mse_loss(predictions[:, 3:], targets[:, 3:])

        # Total loss
        total_loss = vector_loss + mse_loss_gripper

        # Compute deviations in each axis
        deviations = pred_direction_normalized - target_direction_normalized

        return total_loss, deviations

class Trainer:
    def __init__(self, 
                 model, 
                 train_set, 
                 validation_set, 
                 results_path, 
                 model_name, 
                 optimizer_name='adam', 
                 batch_size=32, 
                 epochs=100, 
                 lr=0.001, 
                 momentum=0.9,
                 cosine_similarity_loss=False):
        self.accelerator = Accelerator()
        self.model = model
        self.model_name = model_name
        self.train_set = train_set
        self.validation_set = validation_set
        self.results_path = results_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum  # Momentum parameter
        
        self.aux = True if 'aux' in model_name else False

        self.device = self.accelerator.device
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        self.validation_loader = DataLoader(validation_set, batch_size=self.batch_size, shuffle=False)
        self.criterion = DeltaControlLoss() if cosine_similarity_loss else nn.MSELoss()
        
        # Choose optimizer based on the optimizer_name argument
        self.optimizer = self.get_optimizer(optimizer_name)

        # Learning rate scheduler based on training loss plateau
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        # Prepare the model, optimizer, and dataloaders for distributed training
        self.model, self.optimizer, self.train_loader, self.validation_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.validation_loader
        )

        if not os.path.exists(results_path):
            os.makedirs(results_path)

        # Initialize TensorBoard SummaryWriter
        self.writer = SummaryWriter(log_dir=os.path.join(results_path, 'tensorboard_logs', model_name))
        self.start_epoch = 0  # Initialize start_epoch

    def get_optimizer(self, optimizer_name):
        """Return the optimizer based on the provided optimizer_name."""
        if optimizer_name.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.lr)
        elif optimizer_name.lower() == 'sgd':
            # If SGD is selected, enable momentum by passing the momentum argument
            return optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        elif optimizer_name.lower() == 'rmsprop':
            return optim.RMSprop(self.model.parameters(), lr=self.lr)
        elif optimizer_name.lower() == 'adagrad':
            return optim.Adagrad(self.model.parameters(), lr=self.lr)
        elif optimizer_name.lower() == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from {checkpoint_path}, starting from epoch {self.start_epoch}")

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            self.model.train()
            running_loss = 0.0
            epoch_progress = tqdm(total=len(self.train_loader), desc=f"Epoch [{epoch + 1}/{self.epochs}]", position=0, leave=True)

            validate_every = len(self.train_loader) // 2
            save_model_every = len(self.train_loader) // 2

            for i, (inputs, labels) in enumerate(self.train_loader):
                self.model.train()
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss, deviations = self.criterion(outputs, labels)
                self.accelerator.backward(loss)
                self.optimizer.step()

                running_loss += loss.item()

                if i % 5 == 4:  # Update progress bar every 5 iterations
                    avg_loss = running_loss / 5
                    epoch_progress.set_postfix({'Loss': f'{avg_loss:.4f}'})
                    running_loss = 0.0

                # Log training loss to TensorBoard
                step = epoch * len(self.train_loader) + i
                self.writer.add_scalar('Training Loss', loss.item(), step)

                # Log deviations to TensorBoard
                self.writer.add_scalar('Deviation/X', deviations[:, 0].mean().item(), step)
                self.writer.add_scalar('Deviation/Y', deviations[:, 1].mean().item(), step)
                self.writer.add_scalar('Deviation/Z', deviations[:, 2].mean().item(), step)

                # Log gradients and absolute weights to TensorBoard to monitor vanishing/exploding gradients
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        self.writer.add_histogram(f'Gradients/{name}', param.grad, step)
                    self.writer.add_histogram(f'Weights/{name}', param.data.abs(), step)

                if validate_every != 0 and i % validate_every == validate_every - 1:
                    val_loss, outputs, labels, val_deviations = self.validate()
                    self.model.train()  # Return model to train mode after validation
                    self.save_validation(val_loss, outputs, labels, epoch + 1, i + 1)

                    # Log validation loss and deviations to TensorBoard
                    self.writer.add_scalar('Validation Loss', val_loss, step)
                    self.writer.add_scalar('Validation Deviation/X', val_deviations[0].mean().item(), step)
                    self.writer.add_scalar('Validation Deviation/Y', val_deviations[1].mean().item(), step)
                    self.writer.add_scalar('Validation Deviation/Z', val_deviations[2].mean().item(), step)

                if save_model_every != 0 and i % save_model_every == save_model_every - 1:
                    self.save_model(epoch + 1, i + 1)

                # Update tqdm progress bar
                epoch_progress.update(1)

            # Scheduler step based on training loss
            self.scheduler.step(loss.item())

            # Save checkpoint at the end of each epoch
            self.save_model(epoch + 1, len(self.train_loader))

            epoch_progress.close()

        # Close the TensorBoard writer when training is complete
        self.writer.close()

    def recover_action_vector(self, output, action_vector_channels=7):
        """
        Recovers the action vector from the concatenated output.
        
        Parameters:
            output (torch.Tensor): The concatenated tensor (reconstructed image + action vector)
            action_vector_channels (int): The number of channels in the action vector (7 in this case).
            
        Returns:
            torch.Tensor: The recovered action vector with the original shape [batch_size, 7].
        """
        
        # Step 1: Extract the action vector part from the concatenated output
        # We assume the last 'action_vector_channels' correspond to the action vector
        recovered_action_vector = output[:, -action_vector_channels:, :, :]
        
        # Step 2: Reduce the dimensions by taking the mean over height and width
        # This reduces (batch_size, 7, 128, 128) to (batch_size, 7, 1, 1)
        recovered_action_vector = recovered_action_vector.mean(dim=[2, 3], keepdim=True)
        
        # Step 3: Reshape to the original shape (batch_size, 7)
        batch_size = output.size(0)
        recovered_action_vector = recovered_action_vector.view(batch_size, action_vector_channels)
        
        return recovered_action_vector

    def validate(self):
        self.model.eval()
        total_val_loss = 0.0
        all_deviations = []
        with torch.no_grad():
            for inputs, labels in tqdm(self.validation_loader, desc="Validating", leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                
                if self.aux:
                    outputs = self.recover_action_vector(outputs)
                    labels = self.recover_action_vector(labels)
                    
                loss, deviations = self.criterion(outputs, labels)
                total_val_loss += loss.item()
                all_deviations.append(deviations)

        avg_deviations = torch.cat(all_deviations).mean(dim=0)
        return total_val_loss / len(self.validation_loader), outputs, labels, avg_deviations
    
    def save_validation(self, val_loss, outputs, labels, epoch, iteration, end_of_epoch=False):
        if end_of_epoch:
            with open(os.path.join(self.results_path, f'{self.model_name}_val.csv'), 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([f"Epoch: {epoch} ended after {iteration} iterations, val_loss (MSE): {val_loss}"])
        else:    
            with open(os.path.join(self.results_path, f'{self.model_name}_val.csv'), 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([f"val_loss (MSE): {val_loss}; Epoch: {epoch}; Iteration: {iteration}"])
                num_rows = outputs.shape[0]
                indices = torch.randperm(num_rows)[:10]
                sample_outputs = outputs[indices, :]
                sample_labels = labels[indices, :]
                writer.writerow([f"sample outputs:\n {sample_outputs}"])
                writer.writerow([f"corresponding labels:\n {sample_labels}"])

    def save_model(self, epoch, iteration):
        checkpoint_path = os.path.join(self.results_path, f'{self.model_name}_checkpoint.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

        if isinstance(self.model, ActionExtractionCNN):
            torch.save(self.model.frames_convolution_model.state_dict(), os.path.join(self.results_path, f'{self.model_name}_cnn-{epoch}-{iteration}.pth'))
            torch.save(self.model.action_mlp_model.state_dict(), os.path.join(self.results_path, f'{self.model_name}_mlp-{epoch}-{iteration}.pth'))

        elif isinstance(self.model, ActionExtractionViT):
            torch.save(self.model.frames_convolution_model.state_dict(), os.path.join(self.results_path, f'{self.model_name}_cnn-{epoch}-{iteration}.pth'))
            torch.save(self.model.action_transformer_model.state_dict(), os.path.join(self.results_path, f'{self.model_name}_vit-{epoch}-{iteration}.pth'))

        elif isinstance(self.model, ActionExtractionResNet):
            torch.save(self.model.conv.state_dict(), os.path.join(self.results_path, f'{self.model_name}_resnet-{epoch}-{iteration}.pth'))
            torch.save(self.model.mlp.state_dict(), os.path.join(self.results_path, f'{self.model_name}_mlp-{epoch}-{iteration}.pth'))
            
        elif isinstance(self.model, ResNet3D):
            torch.save(self.model.conv.state_dict(), os.path.join(self.results_path, f'{self.model_name}_resnet-{epoch}-{iteration}.pth'))
            torch.save(self.model.mlp.state_dict(), os.path.join(self.results_path, f'{self.model_name}_mlp-{epoch}-{iteration}.pth'))
            
        elif isinstance(self.model, LatentEncoderPretrainCNNUNet) or isinstance(self.model, LatentEncoderPretrainResNetUNet):
            torch.save(self.model.idm.state_dict(), os.path.join(self.results_path, f'{self.model_name}_idm-{epoch}-{iteration}.pth'))
            torch.save(self.model.fdm.state_dict(), os.path.join(self.results_path, f'{self.model_name}_fdm-{epoch}-{iteration}.pth'))

        elif isinstance(self.model, LatentDecoderMLP):
            torch.save(self.model.mlp.state_dict(), os.path.join(self.results_path, f'{self.model_name}-{epoch}-{iteration}.pth'))

        elif isinstance(self.model, LatentDecoderTransformer):
            torch.save(self.model.transformer.state_dict(), os.path.join(self.results_path, f'{self.model_name}-{epoch}-{iteration}.pth'))

        elif isinstance(self.model, LatentDecoderObsConditionedUNetMLP):
            torch.save(self.model.unet.state_dict(), os.path.join(self.results_path, f'{self.model_name}_unet-{epoch}-{iteration}.pth'))
            torch.save(self.model.mlp.state_dict(), os.path.join(self.results_path, f'{self.model_name}_mlp-{epoch}-{iteration}.pth'))

        elif isinstance(self.model, LatentDecoderAuxiliarySeparateUNetTransformer):
            torch.save(self.model.fdm.state_dict(), os.path.join(self.results_path, f'{self.model_name}_fdm-{epoch}-{iteration}.pth'))
            torch.save(self.model.idm.state_dict(), os.path.join(self.results_path, f'{self.model_name}_idm-{epoch}-{iteration}.pth'))
            torch.save(self.model.transformer.state_dict(), os.path.join(self.results_path, f'{self.model_name}_transformer-{epoch}-{iteration}.pth'))

        elif isinstance(self.model, LatentDecoderAuxiliarySeparateUNetMLP):
            torch.save(self.model.fdm.state_dict(), os.path.join(self.results_path, f'{self.model_name}_fdm-{epoch}-{iteration}.pth'))
            torch.save(self.model.idm.state_dict(), os.path.join(self.results_path, f'{self.model_name}_idm-{epoch}-{iteration}.pth'))
            torch.save(self.model.mlp.state_dict(), os.path.join(self.results_path, f'{self.model_name}_mlp-{epoch}-{iteration}.pth'))

        else:
            print('Model not supported')