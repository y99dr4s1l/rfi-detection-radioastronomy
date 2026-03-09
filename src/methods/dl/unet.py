import time
import numpy as np
import tensorflow as tf


def build_unet(args, n_filters: int = 16, dropout: float = 0.05, batchnorm: bool = True):
    """
    Builds a U-Net model using RFI-NLN's UNET implementation.

    Args:
        args: Args object with input_shape and other model parameters.
        n_filters: Number of filters in the first convolutional layer. Default is 16.
        dropout: Dropout rate. Default is 0.05.
        batchnorm: Whether to use batch normalization. Default is True.

    Returns:
        tf.keras.Model: Compiled U-Net model.
    """
    from models import UNET
    return UNET(args, n_filters=n_filters, dropout=dropout, batchnorm=batchnorm)


def train_unet(
    model,
    train_data: np.ndarray,
    train_masks: np.ndarray,
    epochs: int = 500,
    batch_size: int = 1024,
    learning_rate: float = 1e-4,
    buffer_size: int = 64
):
    """
    Trains a U-Net model for RFI detection.

    Args:
        model: U-Net model instance.
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

        print_epoch('UNET', epoch, time.time() - start, {
            'Loss': loss.numpy(),
            'Accuracy': accuracy.numpy(),
            'Precision': precision.numpy(),
            'Recall': recall.numpy(),
            'F1': f1.numpy()
        }, None)

    return model