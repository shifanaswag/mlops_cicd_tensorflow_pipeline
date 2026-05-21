import numpy as np
import tensorflow as tf

MODEL_PATH = "model/iris_model.keras"

def test_prediction_output():
    model = tf.keras.models.load_model(MODEL_PATH)

    sample_input = np.random.rand(2, 4)
    preds = model.predict(sample_input)

    assert preds.shape[0] == 2


def test_prediction_probability_range():
    model = tf.keras.models.load_model(MODEL_PATH)

    sample_input = np.random.rand(3, 4)
    preds = model.predict(sample_input)

    assert preds.min() >= 0
    assert preds.max() <= 1