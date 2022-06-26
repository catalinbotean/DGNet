import torch

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from datasets import get_training_loader, get_validation_loader
from loss import get_loss_function
from networks import get_network
from optimizers import get_optimizer


class Training:
    def __init__(self):
        self.training_loader = get_training_loader()
        self.validation_loader = get_validation_loader()
        self.network = get_network()
        self.loss_fn = get_loss_function()
        self.optimizer = get_optimizer(self.network)
        self.number_of_epochs = 5

    def train_one_epoch(self, epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.

        for index, training_data in enumerate(self.training_loader):
            inputs, labels = training_data
            self.optimizer.zero_grad()
            outputs = self.network(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

            if index % 1000 == 999:
                last_loss = running_loss / 1000
                print('  batch {} loss: {}'.format(index + 1, last_loss))
                tb_x = epoch_index * len(self.training_loader) + index + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss

    def train(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
        current_epoch = 0
        best_vloss = 1_000_000.
        for epoch in range(self.number_of_epochs):
            print('EPOCH {}:'.format(current_epoch + 1))
            self.network.train(True)
            avg_loss = self.train_one_epoch(current_epoch, writer)
            self.network.train(False)
            running_vloss = 0.0
            for index, validation_data in enumerate(self.validation_loader):
                validation_inputs, validation_labels = validation_data
                validation_outputs = self.network(validation_inputs)
                vloss = self.loss_fn(validation_outputs, validation_labels)
                running_vloss += vloss
            avg_vloss = running_vloss / (index + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
            writer.add_scalars('Training vs. Validation Loss',
                               {'Training': avg_loss, 'Validation': avg_vloss},
                               current_epoch + 1)
            writer.flush()
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                network_path = 'Network_{}_{}'.format(timestamp, current_epoch)
                torch.save(self.network.state_dict(), network_path)
            current_epoch += 1


if __name__ == '__main__':
    Training().train()
