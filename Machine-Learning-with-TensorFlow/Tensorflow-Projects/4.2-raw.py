import pandas as pd
import numpy as np
import tensorflow as tf
import os

# Load data
data = pd.read_csv('data/training.1600000.processed.noemoticon.csv', 
                   encoding='latin-1',
                   header=None)
data.head(3)

# Concatenate the text
text = ' '.join(data[5])
text[:300]

# Vectorization
vocab = sorted(set(text))
print(f'{len(vocab)} unique characters')

char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

# Sequence length and examples per epoch
seq_length = 128
examples_per_epoch = len(text) // (seq_length + 1)

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

# Batch size
BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Define the model function
def get_model(vocab, embedding_dim=256, rnn_units=512):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(len(vocab), embedding_dim, input_shape=[None]),
        tf.keras.layers.LSTM(rnn_units, return_sequences=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(len(vocab))
    ])
    return model

# Initialize the model
model = get_model(vocab)

# Test the model with a batch from the dataset
for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)

sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices])))

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}.weights.h5")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True
)

model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

model.fit(dataset, 
          epochs=20,
          callbacks=[checkpoint_callback],
          verbose=1)

def generate_text(model, start_string):
    num_generate = 1000
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    temperature = 1.0
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])
    return (start_string + ''.join(text_generated))

# Rebuild model with batch size = 1 for generating
generating_model = get_model(vocab)
generating_model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
generating_model.build(tf.TensorShape([1, None]))

# Define start strings to generate text
start_strings = [
    "Once upon a time",
    "In a galaxy far, far away",
    "The quick brown fox",
    "To be or not to be",
    "Hello world"
]

# Generate and print text for each start string
for start_string in start_strings:
    print(f"Start String: {start_string}")
    generated_text = generate_text(generating_model, start_string)
    print(f"Generated Text:\n{generated_text}\n")
    print("="*50)  # Separator for readability
