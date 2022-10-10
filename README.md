# Ex05 : Stock Price Prediction

## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset

Stock price prediction is one among the complex machine learning problems. It dependson a large number of factors which contribute to changes in the supply and demand. 
Stock prices are represented astime series data and neural networks are trained to learn the patterns from trends. Along with thenumerical analysis of the stock trend, this research also considers the textual analysis of it by analyzing the public sentiment from online news sources and blogs. Utilizing both this information, a merged hybrid model is built which can predict the stock trend more accurately. 

## Neural Network Model

![WhatsApp Image 2022-10-10 at 10 06 31 PM](https://user-images.githubusercontent.com/89703145/194915477-029fe61e-0bfe-4067-8e42-2edce10a613d.jpeg)

## DESIGN STEPS

### STEP 1:
Import the required libraries and function which are need

### STEP 2:
Then Upload the required train dataset using the read function and saturate the open price column from the given dataset & use MinMax Scaler to scale the dataset. 

### STEP 3:
Using for loop to geting the x and y train data, create the model and compile it with epochs & follow the same the procedure for test data then scale the predicted data using MinMax Scaler. Finally, Plot the graph with the sufficient data. 

## PROGRAM

```python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from keras.models import Sequential
```

```python3
dataset_train = pd.read_csv('trainset.csv')
dataset_train.head()
train_set = dataset_train.iloc[:,1:2].values
type(train_set)
train_set.shape
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(train_set)
training_set_scaled.shape
X_train_array = []
y_train_array = []
for i in range(60, 1259):
  X_train_array.append(training_set_scaled[i-60:i,0])
  y_train_array.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train_array), np.array(y_train_array)
X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1],1))
X_train.shape
length = 60
n_features = 1
```

```python3
from tensorflow.keras import models
from tensorflow.keras import layers
model = models.Sequential()
model.add(layers.SimpleRNN(60,input_shape=(length,1)))
model.add(layers.Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()
model.fit(X_train1,y_train,epochs=100, batch_size=32)
```

```python3
dataset_test = pd.read_csv('testset.csv')
test_set = dataset_test.iloc[:,1:2].values
test_set.shape
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
inputs = dataset_total.values
inputs = inputs.reshape(-1,1)
inputs_scaled=sc.transform(inputs)
X_test = []
for i in range(60,1384):
  X_test.append(inputs_scaled[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))
X_test.shape
predicted_stock_price_scaled = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price_scaled)
```

```python3
plt.plot(np.arange(0,1384),inputs, color='red', label = 'Test(Real) Google stock price')
plt.plot(np.arange(60,1384),predicted_stock_price, color='blue', label = 'Predicted Google stock price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price of Google')
plt.legend()
plt.show()
```

## OUTPUT

### True Stock Price, Predicted Stock Price vs time

![image](https://user-images.githubusercontent.com/89703145/194909559-b5597dec-dbca-4669-908b-b435d3cee603.png)

### Mean Square Error

![Screenshot (44)](https://user-images.githubusercontent.com/89703145/194912650-b737461e-9578-4102-ae77-3924d95cd324.png)

## RESULT
Thus, the RNN model is created for the stock price prediction and verified successfully.
