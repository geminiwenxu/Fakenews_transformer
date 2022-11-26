import matplotlib.pyplot as plt


def plot(history):
    fig = plt.figure()

    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], color='r', label='train accuracy')
    plt.plot(history['val_acc'], color='g', label='validation accuracy')
    plt.title('Training history')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1.5])

    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], color='b', label='train loss')
    plt.plot(history['val_loss'], color='c', label='validation loss')
    plt.title('Validation history')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1.5])
    plt.savefig("baseline_model.png")