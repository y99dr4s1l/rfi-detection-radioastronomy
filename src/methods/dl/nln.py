"""
methods/dl/nln.py — Nearest Latent Neighbours (Mesarcik 2022)
Build + train (AE+Discriminator adversarial) + inferenza NLN.

Si appoggia ai modelli Autoencoder e Discriminator_x di RFI-NLN/models.py
(passati via sys.path), e a faiss per il nearest-neighbor in latent space.
"""
import time
import numpy as np
import tensorflow as tf
import faiss

# RFI-NLN modules (caricati via sys.path)
from models import Autoencoder, Discriminator_x
from utils.data import patches as patches_module


# ── BUILD ────────────────────────────────────────────────────
def build_nln(args):
    """
    Costruisce autoencoder + discriminator. Ritorna entrambi come tupla.
    Coerente con build_unet/build_rnet che ritornano un singolo modello,
    ma NLN ha intrinsecamente due reti.
    """
    ae            = Autoencoder(args)
    discriminator = Discriminator_x(args)
    return ae, discriminator


# ── TRAIN (AE + adversarial) ─────────────────────────────────
def train_nln(
    ae,
    discriminator,
    ae_train_data: np.ndarray,
    epochs: int       = 50,
    batch_size: int   = 1024,
    buffer_size: int  = 64,
    ae_lr: float      = 1e-4,
    disc_lr: float    = 1e-5,
    gen_lr: float     = 1e-5,
):
    """
    Training adversarial dell'autoencoder.
    ae_train_data deve contenere SOLO patch "normali" (senza RFI),
    coerentemente con il setup anomaly-detection di NLN.
    """
    from utils.training import print_epoch

    ae_optimizer   = tf.keras.optimizers.Adam(ae_lr)
    disc_optimizer = tf.keras.optimizers.Adam(disc_lr)
    gen_optimizer  = tf.keras.optimizers.Adam(gen_lr)
    mse            = tf.keras.losses.MeanSquaredError()

    def ae_recon_loss(x, x_hat):
        return mse(x, x_hat)

    def disc_loss_fn(real_out, fake_out, w=1.0):
        real = mse(tf.ones_like(real_out),  real_out)
        fake = mse(tf.zeros_like(fake_out), fake_out)
        return w * (real + fake)

    def gen_loss_fn(fake_out, w=1.0):
        return w * tf.reduce_mean(mse(tf.ones_like(fake_out), fake_out))

    @tf.function
    def train_step(x):
        with tf.GradientTape() as ae_tape, \
             tf.GradientTape() as disc_tape, \
             tf.GradientTape() as gen_tape:
            x_hat            = ae(x)
            real_out, _      = discriminator(x,     training=True)
            fake_out, _      = discriminator(x_hat, training=True)
            l_ae   = ae_recon_loss(x, x_hat)
            l_disc = disc_loss_fn(real_out, fake_out, 1.0)
            l_gen  = gen_loss_fn(fake_out, 1.0)

        g_ae   = ae_tape.gradient(l_ae,   ae.trainable_variables)
        g_disc = disc_tape.gradient(l_disc, discriminator.trainable_variables)
        g_gen  = gen_tape.gradient(l_gen, ae.decoder.trainable_variables)

        ae_optimizer.apply_gradients(zip(g_ae,    ae.trainable_variables))
        disc_optimizer.apply_gradients(zip(g_disc, discriminator.trainable_variables))
        gen_optimizer.apply_gradients(zip(g_gen,   ae.decoder.trainable_variables))
        return l_ae, l_disc, l_gen

    ds = (
        tf.data.Dataset.from_tensor_slices(ae_train_data)
        .shuffle(buffer_size, seed=42)
        .batch(batch_size)
    )

    for epoch in range(epochs):
        t0 = time.time()
        for batch in ds:
            l_ae, l_disc, l_gen = train_step(batch)
        print_epoch('NLN', epoch, time.time() - t0, {
            'AE Loss'  : l_ae.numpy(),
            'Disc Loss': l_disc.numpy(),
            'Gen Loss' : l_gen.numpy(),
        }, None)

    return ae, discriminator


# ── INFERENZA NLN ────────────────────────────────────────────
def apply_nln(
    ae,
    train_data: np.ndarray,
    test_data: np.ndarray,
    args,
    n_neighbors: int = 2,
    batch_size: int  = 1024,
):
    """
    Calcola NLN error e distanze ricostruite a 512x512.
    Ritorna dict con:
      - nln_error_recon : (N_test_img, 512, 512, 1)
      - dists_recon     : (N_test_img, 512, 512, 1) — distanze interpolate
    """

    def encode_data(encoder, data, bs):
        sample = encoder(data[:1]).numpy()
        latent = sample.shape[-1]
        z = np.empty((len(data), latent), dtype=np.float32)
        ds = tf.data.Dataset.from_tensor_slices(data).batch(bs)
        i = 0
        for b in ds:
            z[i:i+len(b)] = encoder(b).numpy()
            i += len(b)
        return z

    z_train = encode_data(ae.encoder, train_data, batch_size)
    z_query = encode_data(ae.encoder, test_data,  batch_size)

    index = faiss.IndexFlatL2(z_train.shape[1])
    index.add(z_train.astype(np.float32))
    neighbours_dist, neighbours_idx = index.search(
        z_query.astype(np.float32), n_neighbors
    )

    # NLN error: media su K vicini di |test - decode(neighbor_z)|
    test_stacked   = np.stack([test_data] * n_neighbors, axis=1)
    neighbor_preds = ae.predict(
        train_data[neighbours_idx].reshape(-1, *test_data.shape[1:]),
        batch_size=batch_size, verbose=0,
    ).reshape(test_stacked.shape)
    nln_error = np.mean(np.abs(test_stacked - neighbor_preds), axis=1)

    # Ricostruzione 32x32 → 512x512
    nln_error_recon = patches_module.reconstruct(nln_error, args)

    # Distanze ricostruite (interpolate su griglia spaziale)
    from utils.metrics import get_dists
    dists_recon = get_dists(neighbours_dist, args)

    return {
        'nln_error_recon': nln_error_recon,
        'dists_recon'    : dists_recon,
    }