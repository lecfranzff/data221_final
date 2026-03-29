#neural network 

import pandas as pd
import numpy as np 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


#Preprocessing
df = pd.read_csv("C:/Users/alecf/Code/data221_final/cleaned_car_details_dataset.csv")
#Encode 

df_one_hot=df.drop(columns = "Model") #Too complex of a string, if I use one hot could create more than 100+ columns increasing the curse of cardinality
df_one_hot= pd.get_dummies(df_one_hot, columns=["Fuel Type","Transmission","Location","Owner",
                                    "Seller Type","Drivetrain","Make"])

target_Y=df_one_hot.loc[:,"Price"]
feature_X = df_one_hot.loc[:, df_one_hot.columns != "Price"]

train_X, test_X, train_y, test_y = train_test_split(feature_X, target_Y, 
                                        test_size = 0.2,random_state = 1) #Splitting the data into training and testing sets, with 20% of the data reserved for testing. The random_state parameter ensures that the split is reproducible.

columns = train_X.columns.tolist() #Counting how many features I have for the input layer of the neural network
count = 0
for x in columns: 
    count = count + 1 

#Scale
scaler = StandardScaler() #Scale the features to have mean 0 and variance 1, which can help with training the neural network.

X_train_scaled = scaler.fit_transform(train_X)
X_test_scaled = scaler.transform(test_X)

#Building the Model 

model = tf.keras.Sequential([ 
    tf.keras.layers.Input(shape = (X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(512,activation ="relu"), #The first hidden layer has 512 neurons and uses the ReLU activation function. (ReLU performed better than sigmoid)
    tf.keras.layers.Dense(256,activation ="relu"),
    tf.keras.layers.Dense(128,activation ="relu"),
    tf.keras.layers.Dense(64,activation ="relu"),
    tf.keras.layers.Dense(32,activation ="relu"),
    tf.keras.layers.Dense(1)
])

model.summary() #Prints a summary of the model, including the number of parameters in each layer and the total number of parameters in the model.

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
    loss="mse", #Mean Squared Error loss
    metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"),] #Mean Abs Error as a metric to check if model is training well on the epochs
)
