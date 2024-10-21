import sys
import warnings
import argparse
import boto3
import os

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, mpld3
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay  # Pour les jours ouvrables
from tqdm import tqdm
sns.set()
tf.random.set_seed(1234)
from pandas_datareader import data as pdr
import os

import yfinance as yf
yf.pdr_override()

# Symboles de la data sur S3
SYMBOLS_S3 = ['AAPL', 'TSLA', 'GOOGL', 'AMZN', 'MSFT', 'META', 'NVDA']
BUCKET = 'mystockdatapa'  # Ma bucket S3

# Paramètres du modèle
num_layers = 1
size_layer = 128
timestamp = 10
epoch = 300
dropout_rate = 0.8
test_size = 30
learning_rate = 0.01

# Fonction pour récupérer les prédictions stockées dans S3
def fetch_predictions_from_s3(symbol):
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=BUCKET, Key=f'StockPredictions/{symbol}.csv')
    df = pd.read_csv(obj['Body'])
    print(df.columns) 
    return df

# Fonction pour récupérer les dates et les données réelles depuis DataHistory
def get_real_data_from_history(symbol):
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=BUCKET, Key=f'DataHistory/{symbol}.csv')
    df = pd.read_csv(obj['Body'])
    df['Date'] = pd.to_datetime(df['Date'])
    return df[['Date', 'Close']] 

# Fonction pour ajouter des jours ouvrables (pour ne pas compter les week-ends)
def add_business_days(start_date, num_days):
    dates = []
    current_date = start_date
    while len(dates) < num_days:
        current_date += timedelta(days=1)
        # Si ce n'est pas un week-end (lundi = 0, dimanche = 6)
        if current_date.weekday() < 5:  
            dates.append(current_date)
    return dates

# Fonction pour afficher les prédictions et les données réelles avec les tooltips
def display_predictions_with_real_data(predictions, real_data, dates_real, future_dates, symbol):
    plt.figure(figsize=(11, 5))

    # Affichage des prédictions
    all_dates = list(dates_real) + future_dates
    prediction_line, = plt.plot(all_dates, predictions, label=f'{symbol} Predicted Prices', marker='*')
    prediction_labels = [f"""
    <table style="border: 1px solid black; font-weight:bold; font-size:larger; background-color:white">
    <tr style="border: 1px solid black;">
    <th style="border: 1px solid black;">Date:</th>
    <td style="border: 1px solid black;">{date}</td>
    </tr>
    <tr style="border: 1px solid black;">
    <th style="border: 1px solid black;">Predicted Close:</th>
    <td style="border: 1px solid black;">{round(price, 2)}</td>
    </tr>
    </table>
    """ for date, price in zip(all_dates, predictions)]
    
    prediction_tooltip = mpld3.plugins.PointHTMLTooltip(prediction_line, labels=prediction_labels, voffset=10, hoffset=10)
    mpld3.plugins.connect(plt.gcf(), prediction_tooltip)

    # Affichage des données réelles
    real_line, = plt.plot(dates_real, real_data, label='True Trend', c='black', marker='*')
    real_labels = [f"""
    <table style="border: 1px solid black; font-weight:bold; font-size:larger; background-color:white">
    <tr style="border: 1px solid black;">
    <th style="border: 1px solid black;">Date:</th>
    <td style="border: 1px solid black;">{date}</td>
    </tr>
    <tr style="border: 1px solid black;">
    <th style="border: 1px solid black;">True Close:</th>
    <td style="border: 1px solid black;">{round(price, 2)}</td>
    </tr>
    </table>
    """ for date, price in zip(dates_real, real_data)]
    
    real_tooltip = mpld3.plugins.PointHTMLTooltip(real_line, labels=real_labels, voffset=10, hoffset=10)
    mpld3.plugins.connect(plt.gcf(), real_tooltip)

    plt.legend()
    plt.title(f'Stock: {symbol} Predictions and Real Data')
    plt.xticks([])
    plt.autoscale(enable=True, axis='both', tight=None)
    html = mpld3.fig_to_html(plt.gcf())
    return html

# Fonction principale de prédiction
def predict_stock(symbol, period, sim, future):
    # Si le symbole est dans la liste, on récupère directement les prédictions depuis S3
    if symbol in SYMBOLS_S3:
        print(f"Fetching predictions for {symbol} from S3...")
        predictions_df = fetch_predictions_from_s3(symbol)

        # Récupérer les données historiques réelles (dates et prix de clôture) depuis DataHistory
        real_data_df = get_real_data_from_history(symbol)
        dates_real = real_data_df['Date']
        real_data = real_data_df['Close'].values

        # La dernière date des données historiques
        last_date = dates_real.iloc[-1]

        # Ajouter des jours ouvrables pour les futures prédictions (en sautant les week-ends)
        future_dates = add_business_days(last_date, future)  # Ajoute 30 jours ouvrables

        # Affichage des prédictions et des données réelles avec tooltips
        return display_predictions_with_real_data(predictions_df['Predicted_Close'].values, real_data, dates_real, future_dates, symbol)
    
    # Sinon, on exécute le modèle de prédiction
    else:
        print(f"Running prediction model for {symbol}...")
        simulation_size = sim
        test_size = future
        # Télécharger les données boursières
        df = pdr.get_data_yahoo(symbol, period=period, interval="1d")
        df.to_csv('data.csv')
        df = pd.read_csv('data.csv')
        minmax = MinMaxScaler().fit(df.iloc[:, 4:5].astype('float32')) # Close index
        df_log = minmax.transform(df.iloc[:, 4:5].astype('float32')) # Close index
        df_log = pd.DataFrame(df_log)  

        df_train = df_log

        class Model(tf.keras.Model):
            def __init__(self, num_layers, size_layer, output_size, dropout_rate=0.8):
                super(Model, self).__init__()
                self.lstm_layers = [tf.keras.layers.LSTM(size_layer, return_sequences=True, 
                                                         dropout=dropout_rate) for _ in range(num_layers-1)]
                self.lstm_layers.append(tf.keras.layers.LSTM(size_layer))
                self.dense = tf.keras.layers.Dense(output_size)

            def call(self, inputs):
                x = inputs
                for layer in self.lstm_layers:
                    x = layer(x)
                return self.dense(x)

        def calculate_accuracy(real, predict):
            real = np.array(real) + 1
            predict = np.array(predict) + 1
            percentage = 1 - np.sqrt(np.mean(np.square((real - predict) / real)))
            return percentage * 100

        def anchor(signal, weight):
            buffer = []
            last = signal[0]
            for i in signal:
                smoothed_val = last * weight + (1 - weight) * i
                buffer.append(smoothed_val)
                last = smoothed_val
            return buffer

        def forecast():
            model = Model(num_layers, size_layer, df_log.shape[1], dropout_rate)
            optimizer = tf.keras.optimizers.Adam(learning_rate)
            date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()

            pbar = tqdm(range(epoch), desc='train loop')
            for _ in pbar:
                total_loss, total_acc = [], []
                for k in range(0, df_train.shape[0] - 1, timestamp):
                    index = min(k + timestamp, df_train.shape[0] - 1)
                    batch_x = np.expand_dims(df_train.iloc[k: index, :].values, axis=0)
                    batch_y = df_train.iloc[k + 1: index + 1, :].values
                    if batch_y.shape[0] != batch_x.shape[1]:
                        continue
                    
                    with tf.GradientTape() as tape:
                        logits = model(batch_x, training=True)
                        loss = tf.reduce_mean(tf.square(batch_y - logits))
                    
                    grads = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))
                    
                    total_loss.append(loss.numpy())
                    total_acc.append(calculate_accuracy(batch_y[:, 0], logits[:, 0].numpy()))
                pbar.set_postfix(cost=np.mean(total_loss), acc=np.mean(total_acc))
        
            future_day = test_size

            output_predict = np.zeros((df_train.shape[0] + future_day, df_train.shape[1]))
            output_predict[0] = df_train.iloc[0]
            upper_b = (df_train.shape[0] // timestamp) * timestamp

            for k in range(0, (df_train.shape[0] // timestamp) * timestamp, timestamp):
                out_logits = model.predict(np.expand_dims(df_train.iloc[k: k + timestamp], axis=0))
                output_predict[k + 1: k + timestamp + 1] = out_logits

            if upper_b != df_train.shape[0]:
                out_logits = model.predict(np.expand_dims(df_train.iloc[upper_b:], axis=0))
                output_predict[upper_b + 1: df_train.shape[0] + 1] = out_logits
                future_day -= 1
                date_ori.append(date_ori[-1] + timedelta(days=1))

            for i in range(future_day):
                o = output_predict[-future_day - timestamp + i:-future_day + i]
                out_logits = model.predict(np.expand_dims(o, axis=0))
                output_predict[-future_day + i] = out_logits[-1]
                date_ori.append(date_ori[-1] + timedelta(days=1))
        
            output_predict = minmax.inverse_transform(output_predict)
            deep_future = anchor(output_predict[:, 0], 0.4)
        
            return deep_future

        results = []
        for i in range(simulation_size):
            print('Simulation %d:' % (i + 1))
            results.append(forecast())

        date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()
        for i in range(test_size):
            date_ori.append(date_ori[-1] + timedelta(days=1))
        date_ori = pd.Series(date_ori).dt.strftime(date_format='%Y-%m-%d').tolist()

        accepted_results = []
        for r in results:
            if (np.array(r[-test_size:]) < np.min(df['Close'])).sum() == 0 and \
            (np.array(r[-test_size:]) > np.max(df['Close']) * 2).sum() == 0:
                accepted_results.append(r)

        accuracies = [calculate_accuracy(df['Close'].values, r[:-test_size]) for r in accepted_results]

        plt.figure(figsize=(11, 5))
        for no, r in enumerate(accepted_results):
            labels = [f"""
            <table style="border: 1px solid black; font-weight:bold; font-size:larger; background-color:white">
            <tr style="border: 1px solid black;">
            <th style="border: 1px solid black;">Date:</th>
            <td style="border: 1px solid black;">{x}</td>
            </tr>
            <tr style="border: 1px solid black;">
            <th style="border: 1px solid black;">Close:</th>
            <td style="border: 1px solid black;">{round(y,2)}</td>
            </tr>
            </table>
            """ for x, y in zip(date_ori[::5], r[::5])]
            lines = plt.plot(date_ori[::5], r[::5], label='forecast %d' % (no + 1), marker="*")
            tooltips = mpld3.plugins.PointHTMLTooltip(lines[0], labels=labels, voffset=10, hoffset=10)
            mpld3.plugins.connect(plt.gcf(), tooltips)
        lines = plt.plot(df.iloc[:, 0].tolist()[::5], df['Close'][::5], label='true trend', c='black', marker="*")
        labels = [f"""
            <table style="border: 1px solid black; font-weight:bold; font-size:larger; background-color:white">
            <tr style="border: 1px solid black;">
            <th style="border: 1px solid black;">Date:</th>
            <td style="border: 1px solid black;">{y}</td>
            </tr>
            <tr style="border: 1px solid black;">
            <th style="border: 1px solid black;">Close:</th>
            <td style="border: 1px solid black;">{round(x,2)}</td>
            </tr>
            </table>
        """ for x, y in zip(df["Close"].tolist()[::5], df.iloc[:, 0].tolist()[::5])]
        tooltips = mpld3.plugins.PointHTMLTooltip(lines[0], labels=labels, voffset=10, hoffset=10)
        mpld3.plugins.connect(plt.gcf(), tooltips)
        plt.legend()
        plt.title('Stock: %s Average Accuracy: %.4f' % (symbol, np.mean(accuracies)))
        ax = plt.gca()
        ax.set_facecolor("white")
        x_range_future = np.arange(len(results[0]))
        plt.xticks([])
        plt.autoscale(enable=True, axis='both', tight=None)
        os.remove("data.csv")
        html = mpld3.fig_to_html(plt.gcf())
        return html
