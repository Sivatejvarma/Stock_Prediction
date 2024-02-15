from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense

app = Flask(__name__)

@app.route('/')
def home():
   return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    global lstm_model
    if request.method == 'POST':
        code_com = request.form['company_code']
        code_com = code_com.upper()
        df = pd.read_csv(code_com + ".csv")
        df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
        df.index = df['Date']
        plt.plot(df["Close"],label='Close Price history')
        plt.show()
        df['Volume'].plot()
        plt.ylabel('Volume')
        plt.xlabel(None)
        plt.title("Sales Volume")
        plt.show()
        ma_day = [10, 20, 50]
        for ma in ma_day:
            column_name = f"MA for {ma} days"
            df[column_name] = df['Adj Close'].rolling(ma).mean()
        df[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot()
        plt.title('Moving average')
        plt.show()
        data = df.sort_index(ascending=True, axis=0)
        new_dataset = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])
        for i in range(0, len(data)):
            new_dataset["Date"][i] = data['Date'][i]
            new_dataset["Close"][i] = data["Close"][i]
        new_dataset.index = new_dataset.Date
        new_dataset.drop("Date", axis=1, inplace=True)
        final_dataset = new_dataset.values
        train_data = final_dataset[0:987, :]
        valid_data = final_dataset[987:, :]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(final_dataset)
        x_train_data, y_train_data = [], []
        for i in range(60, len(train_data)):
            x_train_data.append(scaled_data[i - 60:i, 0])
            y_train_data.append(scaled_data[i,0])
        x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)
        x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))
        lstm_model = Sequential()
        lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_data.shape[1], 1)))
        lstm_model.add(LSTM(units=50))
        lstm_model.add(Dense(1))
        lstm_model.compile(loss='mean_squared_error', optimizer='adam')
        lstm_model.fit(x_train_data, y_train_data, epochs=1, batch_size=1, verbose=2)
        inputs_data = new_dataset[len(new_dataset) - len(valid_data) - 60:].values
        inputs_data = inputs_data.reshape(-1, 1)
        inputs_data = scaler.transform(inputs_data)
        X_test = []
        for i in range(60, inputs_data.shape[0]):
            X_test.append(inputs_data[i - 60:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        prediction_closing = lstm_model.predict(X_test)
        prediction_closing = scaler.inverse_transform(prediction_closing)
        lstm_model.save("saved_lstm_model.h5")
        train_data = new_dataset[:987]
        valid_data = new_dataset[987:]
        valid_data['Predictions'] = prediction_closing
        plt.plot(train_data["Close"])
        plt.plot(valid_data[["Close", "Predictions"]])
        plt.show()
        output5 = valid_data[["Close", "Predictions"]].to_html()
        return  render_template('index.html',prediction_table1=output5)

if __name__ == "__main__":
   app.run(debug=True)
   
