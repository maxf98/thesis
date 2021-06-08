from env import point_env_vis
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow_probability as tfp

skill_prior = tfp.distributions.Uniform(low=[-1., -1.], high=[1., 1.])

def load_encoder():
    log_dir = "logs/edl/07-06~20-37/discriminator/encoder"
    encoder = keras.models.load_model(log_dir)
    return encoder


def load_decoder():
    log_dir = "logs/edl/07-06~23-03/discriminator/decoder"
    decoder = keras.models.load_model(log_dir)
    return decoder

def discriminator_vis():
    decoder = load_decoder()

    fig, ax = plt.subplots()
    plt.figure(figsize=(7, 7))
    point_env_vis.latent_skill_vis(ax, decoder)

    plt.show()


def inspect_decoder():
    decoder = load_decoder()
    z = skill_prior.sample(10)
    delta_s = decoder.call(z)
    print(delta_s)


def reconstruction():
    encoder, decoder = load_encoder(), load_decoder()
    a = skill_prior.sample(10)
    _, _, z = encoder.call(a)
    ap = decoder.call(z)
    print(a, ap)


if __name__ == "__main__":
    discriminator_vis()
    #inspect_decoder()
    #reconstruction()