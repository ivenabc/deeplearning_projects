import  tensorflow as tf 
from data import Poem
from model import GenModel

data = Poem('../dataset/poetrySong/poetrySongTest.txt')
BATCH_SIZE = 64
BUFFER_SIZE = 10000
checkpoint_dir = './data'

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

# ds = data.get_dataset()
# ds = ds.map(split_input_target)
# ds = ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model = GenModel(data.word_num, loss, batch_size=1)
model = model.model

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

model.summary()


def generate_text(model, start_string):
  num_generate = 80
  input_eval = [data.word_to_id[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)
  text_generated = []
  temperature = 1.0
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      predictions = tf.squeeze(predictions, 0)

      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(data.id_to_word[predicted_id])

  return (start_string + ''.join(text_generated))

print(generate_text(model, start_string=u"山"))