from numpy import array
from keras.models import Sequential
from keras.layers import Dense

X = array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = array([0, 1, 1, 0])
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=2000, verbose=0)
x_input = array([1, 0])
x_input = x_input.reshape((1, 2))
yhat = model.predict(x_input, verbose=0)
print(yhat)