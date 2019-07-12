import  tensorflow as tf 
from data import Poem
from model import GenModel

data = Poem('../dataset/poetrySong/poetrySong.txt')
BATCH_SIZE = 64
BUFFER_SIZE = 10000

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

ds = data.get_dataset()
ds = ds.map(split_input_target)
ds = ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model = GenModel(data.word_num, loss)
model.train(ds)