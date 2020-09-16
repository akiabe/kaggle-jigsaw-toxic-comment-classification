import tensorflow as tf

def baseline_model(VOCAB_SIZE, EMBEDDING_DIM, MAX_LEN):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            VOCAB_SIZE,
            EMBEDDING_DIM,
            input_length=MAX_LEN
        ),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(VOCAB_SIZE, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    return model

def GRULM(VOCAB_SIZE, EMBEDDING_DIM, MAX_LEN):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            VOCAB_SIZE,
            EMBEDDING_DIM,
            input_length=MAX_LEN
        ),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(2)),
        tf.keras.layers.Dense(VOCAB_SIZE, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    return model

def LSTMLM(VOCAB_SIZE, EMBEDDING_DIM, MAX_LEN):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            VOCAB_SIZE,
            EMBEDDING_DIM,
            input_length=MAX_LEN
        ),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(2)),
        tf.keras.layers.Dense(VOCAB_SIZE, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    return model