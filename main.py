import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


carbon_training = pd.read_csv("carbon-training.csv", names=["Brand", "Food Name", "Carbon Footprint"])

carbon_features = carbon_training.copy()
carbon_labels = carbon_training.pop('Carbon Footprint')

carbon_features = np.array(carbon_features)
print(carbon_features)

carbon_model = tf.keras.Sequential([
  layers.Dense(64),
  layers.Dense(1)
])
