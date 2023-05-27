import torch
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_images(image, mean, std):

    image = image.numpy().transpose((1, 2, 0))
    image = std * image + mean
    image = np.clip(image, 0, 1)
    _, ax = plt.subplots(1, 1, figsize=(15, 15))
    ax.axis('off') 
    plt.imshow(image)


class SaveModel:

    def __init__(self, 
                 path_output,
                 best_valid_loss=float('inf')
                 ):
        self.best_valid_loss = best_valid_loss

        os.makedirs(path_output, exist_ok=True)
        self.path_output = path_output
        
    def __call__(self, 
                 current_valid_loss, 
                 current_valid_acc,
                 epoch, 
                 model, 
                 optimizer, 
                 scheduler,
                 dataloader,
                 criterion,
                 index
                 ):
        
        if current_valid_loss < self.best_valid_loss:

            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'loss_epoch': current_valid_loss,
                'acc_epoch': current_valid_acc,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'class_to_idx': dataloader.dataset.class_to_idx,
                'loss': criterion
                }, 
                os.path.join(self.path_output, f'best_model{index}.pth'))
            

    def save_model_final(self,epochs, model, optimizer, scheduler, dataloader, criterion, train_acc, valid_acc, train_loss, valid_loss, index):
        
        print(f"Saving final model...")
        
        torch.save({
                    'epoch': epochs,
                    'model_state': model.state_dict(),
                    'optim_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'class_to_idx': dataloader.dataset.class_to_idx,
                    'loss': criterion,
                    'train_acc': train_acc,
                    'valid_acc': valid_acc,
                    'train_loss': train_loss,
                    'valid_loss': valid_loss
                    }, 
                    os.path.join(self.path_output, f'final_model{index}.pth'))
        

    def save_plots(self, train_acc, valid_acc, train_loss, valid_loss, index):

        plt.figure(figsize=(10, 7))
        plt.plot(train_acc, color='green', linestyle='-', label='train accuracy')
        plt.plot(valid_acc, color='blue', linestyle='-', label='validataion accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy over the epochs')
        plt.savefig(os.path.join(self.path_output, f'accuracy{index}.png'))
        
        
        plt.figure(figsize=(10, 7))
        plt.plot(train_loss, color='orange', linestyle='-', label='train loss')
        plt.plot(valid_loss, color='red', linestyle='-', label='validataion loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss over the epochs')
        plt.savefig(os.path.join(self.path_output, f'loss{index}.png'))
        