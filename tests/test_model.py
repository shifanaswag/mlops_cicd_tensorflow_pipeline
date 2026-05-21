


import tensorflow as tf
import os
import numpy as np

MODEL_PATH = "model/iris_model.keras"

def test_model_exists():
    assert os.path.exists(MODEL_PATH)

def test_model_load():
    model = tf.keras.models.load_model(MODEL_PATH)
    assert model is not None

def test_model_output_shape():
    model = tf.keras.models.load_model(MODEL_PATH)

    sample = np.random.rand(1, 4)
    pred = model.predict(sample)

    assert pred.shape[1] == 3