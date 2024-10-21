import boto3
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import argparse
import os
import sys


# Model parameters
NUM_LAYERS = 1
SIZE_LAYER = 128
TIMESTAMP = 10
EPOCH = 300
DROPOUT_RATE = 0.8
LEARNING_RATE = 0.01


BUCKET = 'mystockdatapa'

class StockPredictor(tf.keras.Model):
    def __init__(self, num_layers, size_layer, output_size, dropout_rate=0.8):
        super(StockPredictor, self).__init__()
        self.lstm_layers = [tf.keras.layers.LSTM(size_layer, return_sequences=True, 
                                                 dropout=dropout_rate) for _ in range(num_layers-1)]
        self.lstm_layers.append(tf.keras.layers.LSTM(size_layer))
        self.dense = tf.keras.layers.Dense(output_size)

    def call(self, inputs):
        x = inputs
        for layer in self.lstm_layers:
            x = layer(x)
        return self.dense(x)

def load_data(symbol):
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=BUCKET, Key=f'DataHistory/{symbol}.csv')
    df = pd.read_csv(obj['Body'])
    df['Date'] = pd.to_datetime(df['Date'])
    return df[['Date', 'Close']]

def preprocess_data(df):
    minmax = MinMaxScaler().fit(df[['Close']].astype('float32'))
    df_log = minmax.transform(df[['Close']].astype('float32'))
    return pd.DataFrame(df_log), minmax

def train_model(df_log):
    model = StockPredictor(NUM_LAYERS, SIZE_LAYER, df_log.shape[1], DROPOUT_RATE)
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

    for _ in range(EPOCH):
        total_loss = []
        for k in range(0, df_log.shape[0] - 1, TIMESTAMP):
            index = min(k + TIMESTAMP, df_log.shape[0] - 1)
            batch_x = np.expand_dims(df_log.iloc[k: index, :].values, axis=0)
            batch_y = df_log.iloc[k + 1: index + 1, :].values
            if batch_y.shape[0] != batch_x.shape[1]:
                continue
            
            with tf.GradientTape() as tape:
                logits = model(batch_x, training=True)
                loss = tf.reduce_mean(tf.square(batch_y - logits))
            
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            total_loss.append(loss.numpy().item()) 

    return model

def forecast(model, df_log, minmax, future_days):
    output_predict = np.zeros((df_log.shape[0] + future_days, df_log.shape[1]))
    output_predict[0] = df_log.iloc[0]
    
    for k in range(0, df_log.shape[0] - 1, TIMESTAMP):
        index = min(k + TIMESTAMP, df_log.shape[0] - 1)
        batch_x = np.expand_dims(df_log.iloc[k: index, :].values, axis=0)
        out_logits = model.predict(batch_x)
        output_predict[k + 1: k + TIMESTAMP + 1] = out_logits

    for i in range(future_days):
        last_sequence = output_predict[-TIMESTAMP-future_days+i:-future_days+i]
        out_logits = model.predict(np.expand_dims(last_sequence, axis=0))
        output_predict[-future_days+i] = out_logits[-1]

    return minmax.inverse_transform(output_predict)[:, 0]

def calculate_performance(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

def save_predictions(predictions, symbol):
    df_predictions = pd.DataFrame(predictions, columns=['Predicted_Close'])
    csv_buffer = df_predictions.to_csv(index=False)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(BUCKET, f'StockPredictions/{symbol}.csv').put(Body=csv_buffer)

def predict_stock(symbol, future_days):
    print(f"Processing stock: {symbol}")
    df = load_data(symbol)
    df_log, minmax = preprocess_data(df)
    
    train_size = int(len(df_log) * 0.8)
    train_data = df_log[:train_size]
    test_data = df_log[train_size:]
    
    model = train_model(train_data)
    
    test_predictions = forecast(model, test_data, minmax, 0)
    
   
    performance = calculate_performance(df['Close'][train_size:].values, test_predictions[:len(test_data)])
    
 
    full_predictions = forecast(model, df_log, minmax, future_days)
    
    save_predictions(full_predictions, symbol)
    print(f"Predictions for {symbol} saved to S3")
    print(f"Performance metrics for {symbol}:")
    for metric, value in performance.items():
        print(f"{metric}: {value}")

def main(future_days):
    symbols = ['AAPL', 'TSLA', 'GOOGL', 'AMZN', 'MSFT', 'META', 'NVDA']
    
    for symbol in symbols:
        predict_stock(symbol, future_days)

if __name__ == "__main__":
    if 'ipykernel' in sys.modules:
        print("Exécution dans Jupyter/IPython")
        FUTURE_DAYS = 30
        main(FUTURE_DAYS)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--future-days', type=int, default=30, help="Nombre de jours futurs à prédire")
        args = parser.parse_args()
        main(args.future_days)