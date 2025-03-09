import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense

# Initialize Spark Session
spark = SparkSession.builder.appName("LinkedInRecommender").getOrCreate()

# Sample user-item interaction data
data = [(0, 0, 4.0), (0, 1, 2.0), (1, 1, 3.5), (1, 2, 5.0), (2, 0, 3.0), (2, 2, 4.5)]
columns = ["user_id", "item_id", "rating"]
df = spark.createDataFrame(data, columns)

# Train ALS Model for collaborative filtering
als = ALS(userCol="user_id", itemCol="item_id", ratingCol="rating", rank=10, maxIter=10, regParam=0.1)
model = als.fit(df)

# Generate predictions
test_df = spark.createDataFrame([(0, 2), (1, 0), (2, 1)], ["user_id", "item_id"])
predictions = model.transform(test_df)
predictions.show()

# Deep Learning Model for Ranking
num_users = 3
num_items = 3

model_dl = Sequential([
    Embedding(input_dim=num_users, output_dim=10, input_length=1),
    Flatten(),
    Dense(10, activation='relu'),
    Dense(1, activation='linear')
])

model_dl.compile(optimizer='adam', loss='mse')

# Dummy data for training
users = np.array([0, 1, 2])
items = np.array([0, 1, 2])
ratings = np.array([4.0, 3.5, 4.5])

model_dl.fit(x=[users, items], y=ratings, epochs=10, verbose=1)

# Predict rankings for new users
predicted_ranks = model_dl.predict([np.array([0, 1, 2]), np.array([2, 0, 1])])
print(predicted_ranks)
