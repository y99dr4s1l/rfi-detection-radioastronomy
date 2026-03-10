import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def RNET(args, dropout: float = 0.05):
    """
    Builds an R-Net model for RFI detection.

    Args:
        args: Args object with input_shape.
        dropout: Dropout rate. Default is 0.05.

    Returns:
        tf.keras.Model: R-Net model.
    """
    input_data = tf.keras.Input(args.input_shape, name='data')

    xp = layers.Conv2D(filters=12, kernel_size=5, strides=(1, 1), padding='same')(input_data)
    x = layers.BatchNormalization()(xp)
    x = layers.Activation('relu')(x)

    x1 = layers.Conv2D(filters=12, kernel_size=5, strides=(1, 1), padding='same')(x)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)
    if dropout > 0:
        x1 = layers.Dropout(dropout)(x1)

    x2 = layers.Conv2D(filters=12, kernel_size=5, strides=(1, 1), padding='same')(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Activation('relu')(x2)

    x3 = x2 + xp
    if dropout > 0:
        x3 = layers.Dropout(dropout)(x3)

    x4 = layers.Conv2D(filters=12, kernel_size=5, strides=(1, 1), padding='same')(x3)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.Activation('relu')(x4)

    x6 = layers.Conv2D(filters=12, kernel_size=5, strides=(1, 1), padding='same')(x4)
    x6 = layers.BatchNormalization()(x6)
    x6 = layers.Activation('relu')(x6)
    if dropout > 0:
        x6 = layers.Dropout(dropout)(x6)

    x7 = x6 + x3

    x8 = layers.Conv2D(filters=12, kernel_size=5, strides=(1, 1), padding='same')(x7)
    x8 = layers.BatchNormalization()(x8)
    x8 = layers.Activation('relu')(x8)

    x_out = layers.Conv2D(
        filters=1, kernel_size=5, strides=(1, 1), padding='same', activation='relu'
    )(x8)

    return tf.keras.Model(inputs=[input_data], outputs=[x_out])


def build_rnet(args, dropout: float = 0.05):
    """
    Wrapper to build R-Net.

    Args:
        args: Args object with input_shape.
        dropout: Dropout rate. Default is 0.05.

    Returns:
        tf.keras.Model: R-Net model.
    """
    return RNET(args, dropout=dropout)


def train_rnet(
    model,
    train_data: np.ndarray,
    train_masks: np.ndarray,
    epochs: int = 500,
    batch_size: int = 1024,
    learning_rate: float = 1e-4,
    buffer_size: int = 64
):
    """
    Trains an R-Net model for RFI detection.

    Args:
        model: R-Net model instance.
        train_data: Training data of shape (n, h, w, 1).
        train_masks: Training masks of shape (n, h, w, 1).
        epochs: Number of training epochs. Default is 500.
        batch_size: Batch size. Default is 1024.
        learning_rate: Learning rate for Adam optimizer. Default is 1e-4.
        buffer_size: Shuffle buffer size. Default is 64.

    Returns:
        Trained model.
    """
    from utils.training import print_epoch

    bce = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    m_precision = tf.keras.metrics.Precision()
    m_recall = tf.keras.metrics.Recall()
    m_accuracy = tf.keras.metrics.BinaryAccuracy()

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            x_hat = model(x, training=True)
            loss = bce(x_hat, y)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        pred = tf.cast(x_hat > 0.5, tf.float32)
        m_precision.update_state(y, pred)
        m_recall.update_state(y, pred)
        m_accuracy.update_state(y, pred)

        precision = m_precision.result()
        recall = m_recall.result()
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

        return loss, m_accuracy.result(), precision, recall, f1

    train_data_ds = (
        tf.data.Dataset.from_tensor_slices(train_data)
        .shuffle(buffer_size, seed=42)
        .batch(batch_size)
    )
    train_masks_ds = (
        tf.data.Dataset.from_tensor_slices(train_masks.astype('float32'))
        .shuffle(buffer_size, seed=42)
        .batch(batch_size)
    )

    for epoch in range(epochs):
        start = time.time()
        for image_batch, mask_batch in zip(train_data_ds, train_masks_ds):
            loss, accuracy, precision, recall, f1 = train_step(image_batch, mask_batch)

        print_epoch('RNET', epoch, time.time() - start, {
            'Loss': loss.numpy(),
            'Accuracy': accuracy.numpy(),
            'Precision': precision.numpy(),
            'Recall': recall.numpy(),
            'F1': f1.numpy()
        }, None)

    return model