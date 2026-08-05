import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
SEQ_LENGTH = 100
BATCH_SIZE = 64
BUFFER_SIZE = 10000
EMBED_DIM = 256
RNN_UNITS = 512
EPOCHS = 5


def load_data():
    """Download and return raw text for training."""
    path = keras.utils.get_file("shakespeare.txt", DATA_URL)
    with open(path, "rb") as f:
        text = f.read().decode("utf-8")
    print(f"Loaded {len(text)} characters of text from {path}")
    return text


def preprocess_data(text):
    """Tokenize the text and prepare training sequences."""
    vocab = sorted(set(text))
    char_to_id = {u: i for i, u in enumerate(vocab)}
    id_to_char = np.array(vocab)

    text_as_int = np.array([char_to_id[c] for c in text], dtype=np.int32)

    # Create sequences of length SEQ_LENGTH + 1 so we can split input and target.
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    sequences = char_dataset.batch(SEQ_LENGTH + 1, drop_remainder=True)

    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    dataset = sequences.map(split_input_target)
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    # Keep a small validation split by taking the last 10% of sequences.
    total_batches = sum(1 for _ in dataset)
    val_batches = max(1, total_batches // 10)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataset_list = list(dataset)
    train_dataset = tf.data.Dataset.from_tensor_slices(dataset_list[:-val_batches])
    val_dataset = tf.data.Dataset.from_tensor_slices(dataset_list[-val_batches:])
    train_dataset = train_dataset.flat_map(lambda x: tf.data.Dataset.from_tensors(x)).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.flat_map(lambda x: tf.data.Dataset.from_tensors(x)).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, vocab, char_to_id, id_to_char


def build_model(vocab_size):
    """Build a simple character-level language model."""
    model = keras.Sequential(
        [
            layers.Embedding(vocab_size, EMBED_DIM, batch_input_shape=[BATCH_SIZE, None]),
            layers.GRU(RNN_UNITS, return_sequences=True, stateful=False),
            layers.Dense(vocab_size),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def train_model(model, train_dataset, val_dataset):
    """Train the model using the prepared datasets."""
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                filepath="rudy_llm_checkpoint.h5",
                save_best_only=True,
                monitor="val_loss",
            )
        ],
    )
    return history


def generate_text(model, start_string, char_to_id, id_to_char, num_generate=500):
    """Generate text from the trained model starting from a prompt."""
    input_eval = [char_to_id[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []
    temperature = 1.0

    for _ in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(id_to_char[predicted_id])

    return start_string + "".join(text_generated)


def evaluate_model(model, char_to_id, id_to_char):
    """Evaluate by generating text and printing model fit metrics."""
    prompt = "ROMEO:"
    generated = generate_text(model, prompt, char_to_id, id_to_char)
    print("\nGenerated sample text:\n")
    print(generated)


def main():
    """Run the full LLM startup workflow end to end."""
    text = load_data()
    train_dataset, val_dataset, vocab, char_to_id, id_to_char = preprocess_data(text)
    model = build_model(len(vocab))
    train_model(model, train_dataset, val_dataset)
    evaluate_model(model, char_to_id, id_to_char)


if __name__ == "__main__":
    main()
