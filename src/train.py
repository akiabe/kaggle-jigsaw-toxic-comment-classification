import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics

import models

def run(fold):
    df = pd.read_csv("../input/train_folds.csv")
    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)

    print("fitting tokenizer...")
    VOCAB_SIZE = 256
    EMBEDDING_DIM = 512
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

    model = models.baseline_model(VOCAB_SIZE, EMBEDDING_DIM, MAX_LEN)

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    print("training model...")

    EPOCHS = 3
    model.fit(
        train_x,
        train_y,
        validation_data=(valid_x, valid_y),
        verbose=1,
        epochs=EPOCHS
    )

if __name__ == "__main__":
    for fold_ in range(3):
        run(fold_)