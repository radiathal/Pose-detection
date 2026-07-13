import tensorflow_hub as hub
import tensorflow as tf

module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
tf.saved_model.save(module, "./movenet_lightning_local", signatures=module.signatures)