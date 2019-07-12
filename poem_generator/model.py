import tensorflow as tf
import os 

def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    
class GenModel(object):
    
    def __init__(self,vocab_size, loss, embedding_dim=256, batch_size=64):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                      batch_input_shape=[batch_size, None]),
            tf.keras.layers.LSTM(1024,
                                 return_sequences=True,
                                 stateful=True,
                                 recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(vocab_size)
        ])

        self.model.compile(optimizer='adam', loss=loss)

    
    def train(self, dataset):
        # Directory where the checkpoints will be saved
        checkpoint_dir = './data'
        # Name of the checkpoint files
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
        checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True)
        EPOCHS=10
        history = self.model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

