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
from action_extractor.architectures.latent_decoders import (
    LatentDecoderMLP,
    LatentDecoderTransformer,
    LatentDecoderObsConditionedUNetMLP,
    LatentDecoderAuxiliarySeparateUNetTransformer,
    LatentDecoderAuxiliarySeparateUNetMLP
)
from action_extractor.architectures.direct_resnet_mlp import ActionExtractionResNet
from action_extractor.architectures.direct_variational_resnet import (
    ActionExtractionVariationalResNet,
    ActionExtractionHypersphericalResNet,
    ActionExtractionSLAResNet
)
from action_extractor.architectures.resnet import ResNet3D
import csv
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
# from utils.utils import check_dataset  # (assuming your own utility function)

# ---------------------------------------------------------------------
# CUSTOM LOSS CLASSES
# ---------------------------------------------------------------------

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

        # Deviations in each axis (for logging)
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

        # MSE loss for last two components
        mse_loss_gripper = F.mse_loss(predictions[:, 3:], targets[:, 3:])

        # Total
        total_loss = vector_loss + mse_loss_gripper

        deviations = torch.abs(pred_direction_normalized - target_direction_normalized)
        return total_loss, deviations

class VAELoss(nn.Module):
    """
    A generic 'variational' style loss with optional warmup/cyclical weighting
    for the KLD. We'll adapt it so it can handle different model latents.
    """
    def __init__(self, reconstruction_loss_fn=None, kld_weight=0.05,
                 schedule_type='warmup', warmup_epochs=10,
                 cycle_length=10, max_weight=0.1, min_weight=0.001):
        super(VAELoss, self).__init__()
        self.reconstruction_loss_fn = (
            reconstruction_loss_fn if reconstruction_loss_fn is not None 
            else nn.MSELoss()
        )
        self.base_kld_weight = kld_weight
        self.eval_kld_weight = kld_weight
        self.schedule_type = schedule_type
        self.current_epoch = 0

        self.warmup_epochs = warmup_epochs
        self.cycle_length = cycle_length
        self.max_weight = max_weight
        self.min_weight = min_weight
        
        self.last_recon_loss = None
        self.last_kld_loss = None
        
    def update_epoch(self, epoch):
        self.current_epoch = epoch
        if self.schedule_type == 'warmup':
            # Linear warmup from 0 to base_kld_weight
            progress = min(1.0, float(self.current_epoch) / float(self.warmup_epochs))
            self.kld_weight = progress * self.base_kld_weight
        elif self.schedule_type == 'cyclical':
            # Cosine annealing between min_weight and max_weight
            cycle_progress = (self.current_epoch % self.cycle_length) / self.cycle_length
            self.kld_weight = self.min_weight + 0.5 * (self.max_weight - self.min_weight) * \
                              (1 + np.cos(cycle_progress * 2 * np.pi))
        else:
            self.kld_weight = self.base_kld_weight

    def forward(self, model, outputs, targets, latents_dict, validation=False):
        # 1) Reconstruction loss
        recon_loss, _ = self.reconstruction_loss_fn(outputs, targets)
        
        # 2) KL from model
        if hasattr(model, 'module'):
            kld_loss = model.module.kl_divergence(**latents_dict)
        else:
            kld_loss = model.kl_divergence(**latents_dict)
        
        # record for logging
        self.last_recon_loss = recon_loss.item()
        self.last_kld_loss = kld_loss.item()

        # Weighted total
        weight = self.eval_kld_weight if validation else self.kld_weight
        total_loss = recon_loss + weight * kld_loss

        # For deviations, assume first 3 channels are direction
        pred_direction = outputs[:, :3]
        target_direction = targets[:, :3]
        pred_dir_norm = F.normalize(pred_direction, dim=1)
        tgt_dir_norm = F.normalize(target_direction, dim=1)
        deviations = torch.abs(pred_dir_norm - tgt_dir_norm)

        return total_loss, deviations

# ---------------------------------------------------------------------
# TRAINER CLASS
# ---------------------------------------------------------------------

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
        
        self.accelerator = Accelerator(mixed_precision='fp16')
        self.model = model
        self.model_name = model_name
        self.train_set = train_set
        self.validation_set = validation_set
        self.results_path = results_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        
        self.aux = True if 'aux' in model_name else False
        self.device = self.accelerator.device

        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        self.validation_loader = DataLoader(validation_set, batch_size=self.batch_size, shuffle=False)

        # Identify the loss type
        self.loss_type = loss.lower()

        # Identify if the model is variational
        if isinstance(self.model, ActionExtractionSLAResNet):
            self.variational_model_type = 'vmf+gaussian'
        elif isinstance(self.model, ActionExtractionHypersphericalResNet):
            self.variational_model_type = 'vmf'
        elif isinstance(self.model, ActionExtractionVariationalResNet):
            self.variational_model_type = 'normal'
        else:
            self.variational_model_type = None

        # Base reconstruction loss
        if self.loss_type == 'mse':
            base_recon_loss = nn.MSELoss()
        elif self.loss_type == 'cosine':
            base_recon_loss = DeltaControlLoss()
        elif self.loss_type == 'cosine+mse':
            base_recon_loss = SumMSECosineLoss()
        else:
            raise ValueError(f"Unknown loss_type {self.loss_type}")
        
        # If model is variational, wrap in VAELoss
        if self.variational_model_type is not None:
            self.criterion = VAELoss(reconstruction_loss_fn=base_recon_loss,
                                     schedule_type='warmup',
                                     warmup_epochs=10)
        else:
            self.criterion = base_recon_loss

        # Optimizer
        self.optimizer = self.get_optimizer(optimizer_name)

        # LR Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        # Prepare for distributed
        self.model, self.optimizer, self.train_loader, self.validation_loader, self.criterion = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.validation_loader, self.criterion
        )

        # Make sure results_path exists
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        self.writer = SummaryWriter(log_dir=os.path.join(results_path, 'tensorboard_logs', model_name))

        self.start_epoch = 0
        self.best_val_loss = float('inf')

    def get_optimizer(self, optimizer_name):
        if optimizer_name.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.lr)
        elif optimizer_name.lower() == 'sgd':
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
            
            epoch_progress = tqdm(
                total=len(self.train_loader), 
                desc=f"Epoch [{epoch + 1}/{self.epochs}]", 
                position=0, 
                leave=True
            )

            # If using VAELoss with a schedule, update the epoch weighting
            if hasattr(self.criterion, 'update_epoch'):
                self.criterion.update_epoch(epoch)
                self.writer.add_scalar('KLD_Weight', self.criterion.kld_weight, epoch)

            for i, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                # Forward
                model_out = self.model(inputs)
                
                # If it's variational
                if self.variational_model_type == 'normal':
                    outputs, mu, logvar = model_out
                    latents_dict = {'mu': mu, 'logvar': logvar}
                    loss, deviations = self.criterion(self.model, outputs, labels, latents_dict)
                elif self.variational_model_type == 'vmf':
                    outputs, mu, kappa = model_out
                    latents_dict = {'mu': mu, 'kappa': kappa}
                    loss, deviations = self.criterion(self.model, outputs, labels, latents_dict)
                elif self.variational_model_type == 'vmf+gaussian':
                    # SLAResNet => possibly (outputs, mu, kappa, c_mu, c_logvar, c)
                    if len(model_out) == 6:
                        outputs, mu, kappa, c_mu, c_logvar, c = model_out
                        latents_dict = {'mu': mu, 'kappa': kappa, 'c_mu': c_mu, 'c_logvar': c_logvar}
                    else:
                        outputs, mu, kappa, c = model_out
                        latents_dict = {'mu': mu, 'kappa': kappa}
                    loss, deviations = self.criterion(self.model, outputs, labels, latents_dict)
                else:
                    # Non-variational
                    outputs = model_out
                    if self.aux:
                        # Some models produce reconstructed images + action vector
                        # so we must slice out the actual 7D action vector, for instance
                        outputs = self.recover_action_vector(outputs)
                        labels = self.recover_action_vector(labels)
                    loss, deviations = self.criterion(outputs, labels)
                
                # Backprop
                self.accelerator.backward(loss)
                self.optimizer.step()

                # Track stats
                running_loss += loss.item()
                running_deviation += deviations.mean().item()

                # Log to TensorBoard
                step = epoch * len(self.train_loader) + i
                self.writer.add_scalar('Training Loss', loss.item(), step)
                self.writer.add_scalar('Deviation/X', deviations[:, 0].mean().item(), step)
                self.writer.add_scalar('Deviation/Y', deviations[:, 1].mean().item(), step)
                self.writer.add_scalar('Deviation/Z', deviations[:, 2].mean().item(), step)

                # If VAELoss, also log reconstruction + KL
                if hasattr(self.criterion, 'last_recon_loss'):
                    self.writer.add_scalar('VAELoss/Reconstruction', self.criterion.last_recon_loss, step)
                    weighted_kld = self.criterion.last_kld_loss * self.criterion.kld_weight
                    self.writer.add_scalar('VAELoss/KLD_Weighted', weighted_kld, step)
                    self.writer.add_scalar('VAELoss/KLD_Raw', self.criterion.last_kld_loss, step)

                # Show progress every log_interval
                log_interval = 20
                if (i + 1) % log_interval == 0 or (i + 1) == len(self.train_loader):
                    avg_loss = running_loss / min(log_interval, (i + 1) % log_interval 
                                                  if (i + 1) % log_interval != 0 else log_interval)
                    avg_deviation = running_deviation / min(log_interval, (i + 1) % log_interval 
                                                            if (i + 1) % log_interval != 0 else log_interval)
                    running_loss = 0.0
                    running_deviation = 0.0

                    epoch_progress.set_postfix({
                        'Loss': f'{avg_loss:.4f}',
                        'Deviation': f'{avg_deviation:.4f}'
                    })

                epoch_progress.update(1)

            epoch_progress.close()

            # ---------------------------------------------------------------------
            # (1) ALWAYS SAVE A "NEWEST" CHECKPOINT AFTER EVERY EPOCH
            # ---------------------------------------------------------------------
            self.save_latest_checkpoint(epoch + 1)

            # Validate
            val_loss, outputs, labels, avg_deviations = self.validate()
            self.save_validation(val_loss, outputs, labels, epoch + 1, i + 1)

            # Log validation metrics
            self.writer.add_scalar('Validation Loss', val_loss, epoch)
            self.writer.add_scalar('Validation Deviation/X', avg_deviations[0].mean().item(), epoch)
            self.writer.add_scalar('Validation Deviation/Y', avg_deviations[1].mean().item(), epoch)
            self.writer.add_scalar('Validation Deviation/Z', avg_deviations[2].mean().item(), epoch)

            # LR schedule
            self.scheduler.step(val_loss)

            # ---------------------------------------------------------------------
            # (2) IF VAL LOSS IMPROVES, SAVE A "BEST" CHECKPOINT + SEPARATE FILES
            # ---------------------------------------------------------------------
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_best_checkpoint(epoch + 1)
                print(f"Validation loss improved to {val_loss:.6f}.  (Best checkpoint saved)")
            else:
                print(f"Validation loss did not improve at epoch {epoch + 1}.")

        # Done with epochs
        self.writer.close()

    def recover_action_vector(self, output, action_vector_channels=7):
        # Example function to slice out the 7D vector from a bigger tensor
        recovered_action_vector = output[:, -action_vector_channels:, :, :]
        recovered_action_vector = recovered_action_vector.mean(dim=[2, 3], keepdim=True)
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

                model_out = self.model(inputs)
                if self.variational_model_type == 'normal':
                    outputs, mu, logvar = model_out
                    latents_dict = {'mu': mu, 'logvar': logvar}
                    loss, deviations = self.criterion(self.model, outputs, labels, latents_dict, validation=True)
                elif self.variational_model_type == 'vmf':
                    outputs, mu, kappa = model_out
                    latents_dict = {'mu': mu, 'kappa': kappa}
                    loss, deviations = self.criterion(self.model, outputs, labels, latents_dict, validation=True)
                elif self.variational_model_type == 'vmf+gaussian':
                    if len(model_out) == 6:
                        outputs, mu, kappa, c_mu, c_logvar, c = model_out
                        latents_dict = {'mu': mu, 'kappa': kappa, 'c_mu': c_mu, 'c_logvar': c_logvar}
                    else:
                        outputs, mu, kappa, c = model_out
                        latents_dict = {'mu': mu, 'kappa': kappa}
                    loss, deviations = self.criterion(self.model, outputs, labels, latents_dict, validation=True)
                else:
                    outputs = model_out
                    if self.aux:
                        outputs = self.recover_action_vector(outputs)
                        labels = self.recover_action_vector(labels)
                    loss, deviations = self.criterion(outputs, labels)

                total_val_loss += loss.item()
                all_deviations.append(deviations)

        avg_val_loss = total_val_loss / len(self.validation_loader)
        avg_val_loss_tensor = torch.tensor(avg_val_loss, device=self.device)
        avg_val_loss_tensor = self.accelerator.gather(avg_val_loss_tensor)
        avg_val_loss = avg_val_loss_tensor.mean().item()

        # Gather the final outputs/labels
        outputs = self.accelerator.gather(outputs)
        labels = self.accelerator.gather(labels)

        avg_deviations = torch.cat(all_deviations).mean(dim=0)
        avg_deviations = self.accelerator.gather(avg_deviations)

        return avg_val_loss, outputs, labels, avg_deviations
    
    def save_validation(self, val_loss, outputs, labels, epoch, iteration, end_of_epoch=False):
        """
        Example CSV logging function. 
        Writes the current val_loss, optionally some sample predictions, etc.
        """
        csv_path = os.path.join(self.results_path, f'{self.model_name}_val.csv')
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if end_of_epoch:
                writer.writerow([f"Epoch: {epoch} ended after {iteration} iterations, val_loss (MSE): {val_loss}"])
            else:
                writer.writerow([f"val_loss (MSE): {val_loss}; Epoch: {epoch}; Iteration: {iteration}"])
                # Optionally log a few samples
                num_rows = outputs.shape[0]
                indices = torch.randperm(num_rows)[:3]  # grab 3 random examples
                sample_outputs = outputs[indices, :]
                sample_labels = labels[indices, :]
                writer.writerow([f"sample outputs:\n {sample_outputs}"])
                writer.writerow([f"corresponding labels:\n {sample_labels}"])
    
    # ---------------------------------------------------------------------
    # SAVE A "LATEST" CHECKPOINT EVERY EPOCH (overwrite each time)
    # ---------------------------------------------------------------------
    def save_latest_checkpoint(self, epoch):
        """
        After every epoch, overwrite a single file that holds the 
        "latest" checkpoint (including epoch, model state, etc.).
        """
        checkpoint_path = os.path.join(self.results_path, f"{self.model_name}_checkpoint_latest.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, checkpoint_path)
        print(f"Saved latest checkpoint (epoch {epoch}) to {checkpoint_path}")

    # ---------------------------------------------------------------------
    # SAVE A "BEST" CHECKPOINT IF VAL LOSS IMPROVES (OVERWRITE OLD BEST)
    # PLUS SEPARATE MODEL FILES, IF DESIRED
    # ---------------------------------------------------------------------
    def save_best_checkpoint(self, epoch):
        """
        Overwrites the single "best" checkpoint file with the current model
        and also saves separate model component files (CNN, MLP, etc.) for
        this best epoch, just like your original scheme.
        """
        # 1) Save "best" checkpoint (single file)
        best_ckpt_path = os.path.join(self.results_path, f"{self.model_name}_checkpoint_best.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, best_ckpt_path)
        print(f"Overwrote best checkpoint with epoch {epoch} at {best_ckpt_path}")

        # 2) (Optional) save separate param files as your original scheme
        #    (If you only want to keep the best epoch’s param files, 
        #     you can remove older ones each time, or keep them. Up to you.)
        #    Here's the same logic from your original save_model() method:
        
        param_file_list = []

        if isinstance(self.model, ActionExtractionCNN):
            cnn_path = f'{self.model_name}_cnn-{epoch}.pth'
            mlp_path = f'{self.model_name}_mlp-{epoch}.pth'
            torch.save(self.model.frames_convolution_model.state_dict(), 
                       os.path.join(self.results_path, cnn_path))
            torch.save(self.model.action_mlp_model.state_dict(), 
                       os.path.join(self.results_path, mlp_path))
            param_file_list.extend([cnn_path, mlp_path])

        elif isinstance(self.model, ActionExtractionViT):
            cnn_path = f'{self.model_name}_cnn-{epoch}.pth'
            vit_path = f'{self.model_name}_vit-{epoch}.pth'
            torch.save(self.model.frames_convolution_model.state_dict(), 
                       os.path.join(self.results_path, cnn_path))
            torch.save(self.model.action_transformer_model.state_dict(), 
                       os.path.join(self.results_path, vit_path))
            param_file_list.extend([cnn_path, vit_path])

        elif isinstance(self.model, ActionExtractionResNet):
            resnet_path = f'{self.model_name}_resnet-{epoch}.pth'
            mlp_path = f'{self.model_name}_mlp-{epoch}.pth'
            torch.save(self.model.conv.state_dict(), os.path.join(self.results_path, resnet_path))
            torch.save(self.model.mlp.state_dict(), os.path.join(self.results_path, mlp_path))
            param_file_list.extend([resnet_path, mlp_path])

        elif isinstance(self.model, ActionExtractionVariationalResNet):
            resnet_path = f'{self.model_name}_resnet-{epoch}.pth'
            fc_mu_path = f'{self.model_name}_fc_mu-{epoch}.pth'
            fc_logvar_path = f'{self.model_name}_fc_logvar-{epoch}.pth'
            mlp_path = f'{self.model_name}_mlp-{epoch}.pth'

            torch.save(self.model.conv.state_dict(), os.path.join(self.results_path, resnet_path))
            torch.save(self.model.fc_mu.state_dict(), os.path.join(self.results_path, fc_mu_path))
            torch.save(self.model.fc_logvar.state_dict(), os.path.join(self.results_path, fc_logvar_path))
            torch.save(self.model.mlp.state_dict(), os.path.join(self.results_path, mlp_path))
            param_file_list.extend([resnet_path, fc_mu_path, fc_logvar_path, mlp_path])

        elif isinstance(self.model, ActionExtractionHypersphericalResNet):
            resnet_path = f'{self.model_name}_resnet-{epoch}.pth'
            fc_mu_path = f'{self.model_name}_fc_mu-{epoch}.pth'
            fc_kappa_path = f'{self.model_name}_fc_kappa-{epoch}.pth'
            mlp_path = f'{self.model_name}_mlp-{epoch}.pth'

            torch.save(self.model.conv.state_dict(), os.path.join(self.results_path, resnet_path))
            torch.save(self.model.fc_mu.state_dict(), os.path.join(self.results_path, fc_mu_path))
            torch.save(self.model.fc_kappa.state_dict(), os.path.join(self.results_path, fc_kappa_path))
            torch.save(self.model.mlp.state_dict(), os.path.join(self.results_path, mlp_path))
            param_file_list.extend([resnet_path, fc_mu_path, fc_kappa_path, mlp_path])

        elif isinstance(self.model, ResNet3D):
            resnet_path = f'{self.model_name}_resnet-{epoch}.pth'
            mlp_path = f'{self.model_name}_mlp-{epoch}.pth'
            torch.save(self.model.conv.state_dict(), os.path.join(self.results_path, resnet_path))
            torch.save(self.model.mlp.state_dict(), os.path.join(self.results_path, mlp_path))
            param_file_list.extend([resnet_path, mlp_path])

        elif isinstance(self.model, LatentEncoderPretrainCNNUNet) or isinstance(self.model, LatentEncoderPretrainResNetUNet):
            idm_path = f'{self.model_name}_idm-{epoch}.pth'
            fdm_path = f'{self.model_name}_fdm-{epoch}.pth'
            torch.save(self.model.idm.state_dict(), os.path.join(self.results_path, idm_path))
            torch.save(self.model.fdm.state_dict(), os.path.join(self.results_path, fdm_path))
            param_file_list.extend([idm_path, fdm_path])

        elif isinstance(self.model, LatentDecoderMLP):
            mlp_path = f'{self.model_name}-{epoch}.pth'
            torch.save(self.model.mlp.state_dict(), os.path.join(self.results_path, mlp_path))
            param_file_list.append(mlp_path)

        elif isinstance(self.model, LatentDecoderTransformer):
            transformer_path = f'{self.model_name}-{epoch}.pth'
            torch.save(self.model.transformer.state_dict(), os.path.join(self.results_path, transformer_path))
            param_file_list.append(transformer_path)

        elif isinstance(self.model, LatentDecoderObsConditionedUNetMLP):
            unet_path = f'{self.model_name}_unet-{epoch}.pth'
            mlp_path = f'{self.model_name}_mlp-{epoch}.pth'
            torch.save(self.model.unet.state_dict(), os.path.join(self.results_path, unet_path))
            torch.save(self.model.mlp.state_dict(), os.path.join(self.results_path, mlp_path))
            param_file_list.extend([unet_path, mlp_path])

        elif isinstance(self.model, LatentDecoderAuxiliarySeparateUNetTransformer):
            fdm_path = f'{self.model_name}_fdm-{epoch}.pth'
            idm_path = f'{self.model_name}_idm-{epoch}.pth'
            transformer_path = f'{self.model_name}_transformer-{epoch}.pth'
            torch.save(self.model.fdm.state_dict(), os.path.join(self.results_path, fdm_path))
            torch.save(self.model.idm.state_dict(), os.path.join(self.results_path, idm_path))
            torch.save(self.model.transformer.state_dict(), os.path.join(self.results_path, transformer_path))
            param_file_list.extend([fdm_path, idm_path, transformer_path])

        elif isinstance(self.model, LatentDecoderAuxiliarySeparateUNetMLP):
            fdm_path = f'{self.model_name}_fdm-{epoch}.pth'
            idm_path = f'{self.model_name}_idm-{epoch}.pth'
            mlp_path = f'{self.model_name}_mlp-{epoch}.pth'
            torch.save(self.model.fdm.state_dict(), os.path.join(self.results_path, fdm_path))
            torch.save(self.model.idm.state_dict(), os.path.join(self.results_path, idm_path))
            torch.save(self.model.mlp.state_dict(), os.path.join(self.results_path, mlp_path))
            param_file_list.extend([fdm_path, idm_path, mlp_path])

        else:
            print('Model type not recognized for separate-file saving.')

        # You could track these param files to remove older ones if you only want the latest best.
        # (Omitted here for brevity.)
