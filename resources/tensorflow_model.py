import tensorflow as tf
import pickle
import csv

BATCH_SIZE = 64
EMBEDDING_DIM = 128 
NEPOCHS=100

tf.keras.backend.set_floatx('float64')
with open('train_dataset_tensors.pickle', 'rb') as fin:
    train_dataset_tensors = pickle.load(fin)

with open('test_dataset_tensors.pickle', 'rb') as fin:
    test_dataset_tensors = pickle.load(fin)

train_dataset = tf.data.Dataset.from_generator(
        lambda: train_dataset_tensors,
        (tf.float64, tf.int32),
        (tf.TensorShape([None, EMBEDDING_DIM]), tf.TensorShape([]))
).shuffle(10000).padded_batch(BATCH_SIZE)
# Use a tf.keras.layers.Masking to automatically ignore padded values

test_dataset = tf.data.Dataset.from_generator(
        lambda: test_dataset_tensors,
        (tf.float64, tf.int32),
        (tf.TensorShape([None, EMBEDDING_DIM]), tf.TensorShape([]))
).batch(1)
model = tf.keras.Sequential((
        tf.keras.layers.Masking(),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.LSTM(256),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation='tanh'),
        tf.keras.layers.Dense(41, activation='softmax'),
))

model.compile(optimizer="adam", loss=tf.keras.losses.sparse_categorical_crossentropy, metrics="acc")
try:
    model.fit(train_dataset, epochs=NEPOCHS)
except KeyboardInterrupt:
    pass

preds = []
for batch in test_dataset:
    content, content_id = batch
    author_p = model.predict(content).squeeze(axis=0)
    author_p = tf.math.argmax(author_p, axis=-1)
    preds.append((content_id.numpy().item(), author_p.numpy().item()))

with open('preds.csv', 'w', newline='') as f:
    fout = csv.writer(f)
    fout.writerows(preds)
