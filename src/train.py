import mlflow
import mlflow.tensorflow
import tensorflow as tf
import os
import joblib

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Create MLflow experiment
mlflow.set_experiment("Iris_Classification")


# Start MLflow tracking
with mlflow.start_run():

    # Load dataset
    df = load_iris()

    # Features and target
    x = df.data
    y = df.target

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42
    )

    # Feature scaling
    scaler = StandardScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    joblib.dump(scaler, "model/scaler.pkl")

    # Build model
    model = Sequential([
        Dense(
            units=16,
            activation='relu',
            input_shape=(4,)
        ),

        Dense(
            units=8,
            activation='relu'
        ),

        Dense(
            units=3,
            activation='softmax'
        )
    ])

    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train model
    history = model.fit(
        x_train,
        y_train,
        epochs=50,
        validation_split=0.3
    )

    # Evaluate model
    loss, accuracy = model.evaluate(
        x_test,
        y_test
    )

    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}")

    # Log parameters
    mlflow.log_param("epochs", 50)
    mlflow.log_param("optimizer", "adam")
    mlflow.log_param("hidden_layer_1", 16)
    mlflow.log_param("hidden_layer_2", 8)

    # Log metrics
    mlflow.log_metric("loss", loss)
    mlflow.log_metric("accuracy", accuracy)

    # Log model to MLflow
    mlflow.tensorflow.log_model(
        model,
        artifact_path="iris_model",
        registered_model_name="Iris_model"
    )
    # Create folders if they don't exist
    os.makedirs("model", exist_ok=True)
    os.makedirs("serving_model", exist_ok=True)
    # Save Keras format (for tests + CI)
    model.save("model/iris_model.keras")
    # Save TF Serving format (for deployment)
    model.export("serving_model/iris_model/1")
    print("\nModel saved successfully!")
    print("Model logged to MLflow successfully!")