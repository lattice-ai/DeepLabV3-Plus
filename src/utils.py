import tensorflow as tf
from matplotlib import pyplot as plt


def plot_samples_matplotlib(display_list):
    plt.figure(figsize=(18, 18))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        if i < 2:
            plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        else:
            plt.imshow(display_list[i])
        plt.axis('off')
    plt.show()
