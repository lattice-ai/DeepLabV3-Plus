import numpy as np
import tensorflow as tf


def read_image(image_file, image_size, is_mask=False):
    image = tf.io.read_file(image_file)
    if is_mask:
        image = tf.image.decode_png(image, channels=1)
        image.set_shape([None, None, 1])
    else:
        image = tf.image.decode_png(image, channels=3)
        image.set_shape([None, None, 3])
    image = (tf.image.resize(images=image, size=image_size))
    image = tf.cast(image, tf.float32) / 127.5 - 1
    return image


def infer(model_file, image_tensor):
    model = tf.keras.models.load_model(model_file)
    predictions = model.predict(np.expand_dims((image_tensor), axis=0))
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=2)
    return predictions
