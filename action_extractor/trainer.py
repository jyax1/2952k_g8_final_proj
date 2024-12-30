import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from action_extractor.architectures.direct_cnn_mlp import ActionExtractionCNN
from action_extractor.architectures.direct_cnn_vit import ActionExtractionViT
from action_extractor.architectures.latent_encoders import LatentEncoderPretrainCNNUNet, LatentEncoderPretrainResNetUNet
from action_extractor.architectures.latent_decoders import *
from action_extractor.architectures.direct_resnet_mlp import ActionExtractionResNet
from action_extractor.architectures.direct_variational_resnet import ActionExtractionVariationalResNet, ActionExtractionHypersphericalResNet
from action_extractor.architectures.resnet import ResNet3D
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

        # Compute absolute deviations in each axis
        deviations = torch.abs(pred_direction_normalized - target_direction_normalized)

        return total_loss, deviations
    
class SumMSECosineLoss(nn.Module):
    def __init__(self):
        super(SumMSECosineLoss, self).__init__()

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
        vector_loss = direction_loss + magnitude_loss

        # Compute MSE loss for the last two components
        mse_loss_gripper = 2 * F.mse_loss(predictions[:, 3:], targets[:, 3:])

        # Total loss
        total_loss = vector_loss + mse_loss_gripper

        # Compute absolute deviations in each axis
        deviations = torch.abs(pred_direction_normalized - target_direction_normalized)

        return total_loss, deviations

class VAELoss(nn.Module):
    def __init__(self, reconstruction_loss_fn=None, kld_weight=0.05,
                 schedule_type='constant', warmup_epochs=10,
                 cycle_length=10, max_weight=0.1, min_weight=0.001):
        super(VAELoss, self).__init__()
        self.reconstruction_loss_fn = reconstruction_loss_fn if reconstruction_loss_fn is not None else nn.MSELoss()
        self.base_kld_weight = kld_weight
        self.eval_kld_weight = kld_weight
        self.schedule_type = schedule_type
        self.current_epoch = 0
        
        # Warmup parameters
        self.warmup_epochs = warmup_epochs
        
        # Cyclical parameters
        self.cycle_length = cycle_length
        self.max_weight = max_weight
        self.min_weight = min_weight
        
        self.last_recon_loss = None
        self.last_kld_loss = None
        
    def update_epoch(self, epoch):
        """Update current epoch and kld_weight"""
        self.current_epoch = epoch
        
        if self.schedule_type == 'warmup':
            # Linear warmup from 0 to base_kld_weight
            self.kld_weight = min(1.0, self.current_epoch / self.warmup_epochs) * self.base_kld_weight
            
        elif self.schedule_type == 'cyclical':
            # Cosine annealing between min_weight and max_weight
            cycle_progress = (self.current_epoch % self.cycle_length) / self.cycle_length
            self.kld_weight = self.min_weight + 0.5 * (self.max_weight - self.min_weight) * \
                            (1 + np.cos(cycle_progress * 2 * np.pi))
        else:
            self.kld_weight = self.base_kld_weight

    def forward(self, model, outputs, targets, mu, distribution_param, validation=False):
        # Reconstruction loss
        recon_loss, _ = self.reconstruction_loss_fn(outputs, targets)
        
        # KL divergence using model's implementation
        kld_loss = model.kl_divergence(mu, distribution_param)
        
        # Save losses for logging
        self.last_recon_loss = recon_loss.item()
        self.last_kld_loss = kld_loss.item()

        # Total loss
        weight = self.eval_kld_weight if validation else self.kld_weight
        total_loss = recon_loss + weight * kld_loss

        # Compute deviations similar to other loss functions
        # Assume the first three components are direction vectors
        pred_direction = outputs[:, :3]
        target_direction = targets[:, :3]

        # Normalize direction vectors
        pred_direction_normalized = F.normalize(pred_direction, dim=1)
        target_direction_normalized = F.normalize(target_direction, dim=1)

        # Calculate deviations
        deviations = torch.abs(pred_direction_normalized - target_direction_normalized)

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
                 loss='mse',
                 vae=False,
                 num_gpus=1):
        # Initialize accelerator with correct device configuration
        self.accelerator = Accelerator(
            n_processes=num_gpus if num_gpus else None,
            mixed_precision='no'
        )
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
        self.loss_type = loss.lower()
        
        if self.loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif self.loss_type == 'cosine':
            self.criterion = DeltaControlLoss()
        elif self.loss_type == 'cosine+mse':
            self.criterion = SumMSECosineLoss()
            
        self.vae = vae
        if vae:
            self.criterion = VAELoss(reconstruction_loss_fn=self.criterion,
                                     schedule_type='warmup', warmup_epochs=10)
        
        # Choose optimizer based on the optimizer_name argument
        self.optimizer = self.get_optimizer(optimizer_name)

        # Learning rate scheduler based on training loss plateau
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        # Prepare the model, optimizer, and dataloaders for distributed training
        self.model, self.optimizer, self.train_loader, self.validation_loader, self.criterion = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.validation_loader, self.criterion
        )

        if not os.path.exists(results_path):
            os.makedirs(results_path)

        # Initialize TensorBoard SummaryWriter
        self.writer = SummaryWriter(log_dir=os.path.join(results_path, 'tensorboard_logs', model_name))
        self.start_epoch = 0  # Initialize start_epoch
        self.best_val_loss = float('inf')  # Initialize with infinity

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
            running_deviation = 0.0
            epoch_progress = tqdm(total=len(self.train_loader), desc=f"Epoch [{epoch + 1}/{self.epochs}]", position=0, leave=True)
            
            if self.vae:
                self.criterion.update_epoch(epoch)
                self.writer.add_scalar('KLD_Weight', self.criterion.kld_weight, epoch)

            for i, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                
                if self.vae:
                    outputs, mu, distribution_param = self.model(inputs)
                    loss, deviations = self.criterion(self.model, outputs, labels, mu, distribution_param)
                else:
                    outputs = self.model(inputs)
                    loss, deviations = self.criterion(outputs, labels)
                    
                self.accelerator.backward(loss)
                self.optimizer.step()

                running_loss += loss.item()
                running_deviation += deviations.mean().item()

                # Log training loss and deviations to TensorBoard every iteration
                step = epoch * len(self.train_loader) + i
                self.writer.add_scalar('Training Loss', loss.item(), step)
                self.writer.add_scalar('Deviation/X', deviations[:, 0].mean().item(), step)
                self.writer.add_scalar('Deviation/Y', deviations[:, 1].mean().item(), step)
                self.writer.add_scalar('Deviation/Z', deviations[:, 2].mean().item(), step)
                
                if self.vae:
                    self.writer.add_scalar('VAELoss/Reconstruction', self.criterion.last_recon_loss, step)
                    self.writer.add_scalar('VAELoss/KLD_Weighted', self.criterion.last_kld_loss * self.criterion.kld_weight, step)
                    self.writer.add_scalar('VAELoss/KLD_Raw', self.criterion.last_kld_loss, step)

                # Log weights and gradients to TensorBoard every 5 iterations
                if (i + 1) % 5 == 0:
                    # Average loss and deviation over the last 5 iterations
                    avg_loss = running_loss / 5
                    avg_deviation = running_deviation / 5
                    running_loss = 0.0  # Reset running loss
                    running_deviation = 0.0  # Reset running deviation

                    # Update progress bar
                    epoch_progress.set_postfix({'Loss': f'{avg_loss:.4f}', 'Deviation': f'{avg_deviation:.4f}'})

                    # Log weights and gradients
                    for name, param in self.model.named_parameters():
                        if param.requires_grad:
                            # Log weights
                            self.writer.add_histogram(f'Weights/{name}', param.data.cpu().numpy(), step)
                            # Log gradients
                            if param.grad is not None:
                                self.writer.add_histogram(f'Gradients/{name}', param.grad.data.cpu().numpy(), step)

                epoch_progress.update(1)

            epoch_progress.close()

            # Perform validation after each epoch
            val_loss, outputs, labels, avg_deviations = self.validate()

            self.save_validation(val_loss, outputs, labels, epoch + 1, i + 1)

            # Log validation loss and deviations to TensorBoard
            self.writer.add_scalar('Validation Loss', val_loss, epoch)
            self.writer.add_scalar('Validation Deviation/X', avg_deviations[0].mean().item(), epoch)
            self.writer.add_scalar('Validation Deviation/Y', avg_deviations[1].mean().item(), epoch)
            self.writer.add_scalar('Validation Deviation/Z', avg_deviations[2].mean().item(), epoch)

            # Learning rate scheduler step based on validation loss
            self.scheduler.step(val_loss)

            # Save the model if validation loss has decreased
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model(epoch + 1)
                print(f"Validation loss improved to {val_loss:.6f}. Model saved at epoch {epoch + 1}.")
            else:
                print(f"Validation loss did not improve at epoch {epoch + 1}.")

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
                if self.vae:
                    outputs, mu, distribution_param = self.model(inputs)
                    loss, deviations = self.criterion(self.model, outputs, labels, mu, distribution_param, validation=True)
                else:
                    outputs = self.model(inputs)
                    if self.aux:
                        outputs = self.recover_action_vector(outputs)
                        labels = self.recover_action_vector(labels)
                    loss, deviations = self.criterion(outputs, labels)

                total_val_loss += loss.item()
                all_deviations.append(deviations.cpu())

        avg_val_loss = total_val_loss / len(self.validation_loader)
        avg_deviations = torch.cat(all_deviations).mean(dim=0)
        
        avg_val_loss = self.accelerator.gather(avg_val_loss).mean()
        outputs = self.accelerator.gather(outputs)
        labels = self.accelerator.gather(labels)
        avg_deviations = self.accelerator.gather(avg_deviations)
        
        return avg_val_loss, outputs, labels, avg_deviations
    
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

    def save_model(self, epoch):
        checkpoint_path = os.path.join(self.results_path, f'{self.model_name}_checkpoint.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

        # Save model components based on the model type
        if isinstance(self.model, ActionExtractionCNN):
            torch.save(self.model.frames_convolution_model.state_dict(), os.path.join(self.results_path, f'{self.model_name}_cnn-{epoch}.pth'))
            torch.save(self.model.action_mlp_model.state_dict(), os.path.join(self.results_path, f'{self.model_name}_mlp-{epoch}.pth'))

        elif isinstance(self.model, ActionExtractionViT):
            torch.save(self.model.frames_convolution_model.state_dict(), os.path.join(self.results_path, f'{self.model_name}_cnn-{epoch}.pth'))
            torch.save(self.model.action_transformer_model.state_dict(), os.path.join(self.results_path, f'{self.model_name}_vit-{epoch}.pth'))

        elif isinstance(self.model, ActionExtractionResNet):
            torch.save(self.model.conv.state_dict(), os.path.join(self.results_path, f'{self.model_name}_resnet-{epoch}.pth'))
            torch.save(self.model.mlp.state_dict(), os.path.join(self.results_path, f'{self.model_name}_mlp-{epoch}.pth'))
            
        elif isinstance(self.model, ActionExtractionVariationalResNet):
            torch.save(self.model.conv.state_dict(), os.path.join(self.results_path, f'{self.model_name}_resnet-{epoch}.pth'))
            torch.save(self.model.fc_mu.state_dict(), os.path.join(self.results_path, f'{self.model_name}_fc_mu-{epoch}.pth'))
            torch.save(self.model.fc_logvar.state_dict(), os.path.join(self.results_path, f'{self.model_name}_fc_logvar-{epoch}.pth'))
            torch.save(self.model.mlp.state_dict(), os.path.join(self.results_path, f'{self.model_name}_mlp-{epoch}.pth'))
            
        elif isinstance(self.model, ActionExtractionHypersphericalResNet):
            torch.save(self.model.conv.state_dict(), os.path.join(self.results_path, f'{self.model_name}_resnet-{epoch}.pth'))
            torch.save(self.model.fc_mu.state_dict(), os.path.join(self.results_path, f'{self.model_name}_fc_mu-{epoch}.pth'))
            torch.save(self.model.fc_kappa.state_dict(), os.path.join(self.results_path, f'{self.model_name}_fc_kappa-{epoch}.pth'))
            torch.save(self.model.mlp.state_dict(), os.path.join(self.results_path, f'{self.model_name}_mlp-{epoch}.pth'))
            
        elif isinstance(self.model, ResNet3D):
            torch.save(self.model.conv.state_dict(), os.path.join(self.results_path, f'{self.model_name}_resnet-{epoch}.pth'))
            torch.save(self.model.mlp.state_dict(), os.path.join(self.results_path, f'{self.model_name}_mlp-{epoch}.pth'))
            
        elif isinstance(self.model, LatentEncoderPretrainCNNUNet) or isinstance(self.model, LatentEncoderPretrainResNetUNet):
            torch.save(self.model.idm.state_dict(), os.path.join(self.results_path, f'{self.model_name}_idm-{epoch}.pth'))
            torch.save(self.model.fdm.state_dict(), os.path.join(self.results_path, f'{self.model_name}_fdm-{epoch}.pth'))

        elif isinstance(self.model, LatentDecoderMLP):
            torch.save(self.model.mlp.state_dict(), os.path.join(self.results_path, f'{self.model_name}-{epoch}.pth'))

        elif isinstance(self.model, LatentDecoderTransformer):
            torch.save(self.model.transformer.state_dict(), os.path.join(self.results_path, f'{self.model_name}-{epoch}.pth'))

        elif isinstance(self.model, LatentDecoderObsConditionedUNetMLP):
            torch.save(self.model.unet.state_dict(), os.path.join(self.results_path, f'{self.model_name}_unet-{epoch}.pth'))
            torch.save(self.model.mlp.state_dict(), os.path.join(self.results_path, f'{self.model_name}_mlp-{epoch}.pth'))

        elif isinstance(self.model, LatentDecoderAuxiliarySeparateUNetTransformer):
            torch.save(self.model.fdm.state_dict(), os.path.join(self.results_path, f'{self.model_name}_fdm-{epoch}.pth'))
            torch.save(self.model.idm.state_dict(), os.path.join(self.results_path, f'{self.model_name}_idm-{epoch}.pth'))
            torch.save(self.model.transformer.state_dict(), os.path.join(self.results_path, f'{self.model_name}_transformer-{epoch}.pth'))

        elif isinstance(self.model, LatentDecoderAuxiliarySeparateUNetMLP):
            torch.save(self.model.fdm.state_dict(), os.path.join(self.results_path, f'{self.model_name}_fdm-{epoch}.pth'))
            torch.save(self.model.idm.state_dict(), os.path.join(self.results_path, f'{self.model_name}_idm-{epoch}.pth'))
            torch.save(self.model.mlp.state_dict(), os.path.join(self.results_path, f'{self.model_name}_mlp-{epoch}.pth'))

        else:
            print('Model not supported')