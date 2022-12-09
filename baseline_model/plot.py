import matplotlib.pyplot as plt
import torch


def plot(history):
    plt.subplot(1, 2, 1)
    plt.plot(torch.tensor(history['train_acc']).cpu(), color='r', label='train accuracy')
    plt.plot(torch.tensor(history['val_acc']).cpu(), color='g', label='validation accuracy')
    plt.title('Accuray')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1.5])

    plt.subplot(1, 2, 2)
    plt.plot(torch.tensor(history['train_loss']).cpu(), color='b', label='train loss')
    plt.plot(torch.tensor(history['val_loss']).cpu(), color='c', label='validation loss')
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1.5])
    plt.tight_layout()
    plt.savefig("model.png")
