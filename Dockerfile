FROM tensorflow/serving
COPY serving_model/iris_model /models/iris_model

ENV MODEL_NAME=iris_model
EXPOSE 8501

