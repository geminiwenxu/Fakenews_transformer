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
    plt.ylim([0, 1])

    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], color='b', label='train loss')
    plt.plot(history['val_loss'], color='c', label='validation loss')
    plt.title('Validation history')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1.5])
    plt.savefig("baseline_model.png")

    # plt.show()
    #
    #
    # ax1 = plt.axes()
    # ax1.plot(history['train_acc'], color='r', label='train accuracy')
    # ax1.plot(history['val_acc'], color='g', label='validation accuracy')
    # ax1.set_title('Training history')
    # ax1.ylabel('Accuracy')
    # ax1.xlabel('Epoch')
    # ax1.legend(['train', 'val'], loc='upper left')
    # ax1.ylim([0, 1.5])
    # ax1.savefig("baseline_model_accuracy.png")
    #
    # ax2 = plt.axes()
    # ax2.plot(history['train_loss'], color='b', label='train loss')
    # ax2.plot(history['val_loss'], color='c', label='validation loss')
    # ax2.set_title('Validation history')
    # ax2.ylabel('Loss')
    # ax2.xlabel('Epoch')
    # ax2.legend()
    # ax2.ylim([0, 1.5])
    # ax2.savefig("baseline_model_loss.png")
