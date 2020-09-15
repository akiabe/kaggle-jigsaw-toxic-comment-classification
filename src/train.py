import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics

def run(fold):
    df = pd.read_csv("../input/train_folds.csv")
    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)

    print("fitting tokenizer...")

    VOCAB_SIZE = 10000
    EMBEDDING_DIM = 16
    MAX_LEN = 128
    TRUNCATING = "post"
    OOV_TOKEN = "<OOV>"

    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=VOCAB_SIZE,
        oov_token=OOV_TOKEN
    )

    tokenizer.fit_on_texts(df.comment_text.values.tolist())

    train_x = tokenizer.texts_to_sequences(train_df.comment_text.values)
    valid_x = tokenizer.texts_to_sequences(valid_df.comment_text.values)

    train_x = tf.keras.preprocessing.sequence.pad_sequences(
        train_x,
        maxlen=MAX_LEN,
        truncating=TRUNCATING
    )
    valid_x = tf.keras.preprocessing.sequence.pad_sequences(
        valid_x,
        maxlen=MAX_LEN
    )

    train_y = train_df.toxic.values
    valid_y = valid_df.toxic.values

    print("loading model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            VOCAB_SIZE,
            EMBEDDING_DIM,
            input_length=MAX_LEN
        ),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(6,activation="relu"),
        tf.keras.layers.Dense(1,activation="sigmoid")
    ])

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    print("training model...")
    EPOCHS = 10
    model.fit(
        train_x,
        train_y,
        validation_data=(valid_x,valid_y),
        verbose=1,
        epochs=EPOCHS
    )

    valid_preds = model.predict(valid_x)
    valid_preds = np.array(valid_preds) >= 0.5
    accuracy = metrics.accuracy_score(valid_y, valid_preds)
    print(f"Accuracy Score = {accuracy}")

if __name__ == "__main__":
    run(fold=0)
    run(fold=1)
    run(fold=2)