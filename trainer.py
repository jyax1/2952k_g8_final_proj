import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from accelerate import Accelerator
from models.direct_cnn import ActionExtractionCNN
import csv

class Trainer:
    def __init__(self, model, train_set, validation_set, results_path, model_name, batch_size=32, epochs=10, lr=0.001):
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
            for i, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                self.accelerator.backward(loss)
                self.optimizer.step()

                running_loss += loss.item()
                if i % 10 == 9:  # Print every 10 batches
                    print(f'Epoch [{epoch + 1}/{self.epochs}], Step [{i + 1}/{len(self.train_loader)}], Loss: {running_loss / 10:.4f}')
                    running_loss = 0.0

            # Validate at the end of each epoch
            val_loss = self.validate()
            print(f'Epoch [{epoch + 1}/{self.epochs}] ended, Validation Loss: {val_loss:.4f}')
            self.val_losses.append(val_loss)

            # Save model after each epoch
            self.save_model(epoch + 1)

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
            with open('val_loss.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for value in self.val_losses:
                    writer.writerow([value])
        else:
            print('Model not supported')