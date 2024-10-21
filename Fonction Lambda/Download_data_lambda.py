import os
import sys
import subprocess

# Fonction pour installer les bibliothèques au moment de l'exécution
def install_dependencies():
    # Installation de yfinance et pandas dans le répertoire temporaire (/tmp/)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "--target", "/tmp/"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "--target", "/tmp/"])

    # Ajouter /tmp/ au chemin d'exécution Python
    sys.path.append("/tmp/")

# Appel de la fonction pour installer les bibliothèques
install_dependencies()

# Import des bibliothèques installées dynamiquement
import yfinance as yf
import pandas as pd
import boto3
from io import StringIO
from datetime import datetime, timedelta

# Configuration AWS
s3_bucket_name = 'mystockdatapa'
s3_folder = 'DataHistory/'
s3_client = boto3.client('s3')

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']

def download_new_stock_data(ticker, start_date):
    """Télécharger les nouvelles données boursières à partir d'une date donnée."""
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date)
    return df

def upload_to_s3(df, ticker):
    """Sauvegarder le DataFrame dans un fichier CSV et l'envoyer à S3."""
    csv_buffer = StringIO()
    df.to_csv(csv_buffer)
    s3_key = f'{s3_folder}{ticker}.csv'
    s3_client.put_object(Bucket=s3_bucket_name, Key=s3_key, Body=csv_buffer.getvalue())

def download_from_s3(ticker):
    """Télécharger le fichier CSV existant de S3."""
    s3_key = f'{s3_folder}{ticker}.csv'
    try:
        obj = s3_client.get_object(Bucket=s3_bucket_name, Key=s3_key)
        df = pd.read_csv(obj['Body'])
        return df
    except s3_client.exceptions.NoSuchKey:
        return pd.DataFrame()  # Retourne un DataFrame vide si le fichier n'existe pas encore

def update_stock_data():
    """Mise à jour des données boursières et stockage sur S3."""
    for ticker in tickers:
        print(f"Updating data for {ticker}...")

        # Télécharger les données existantes depuis S3
        existing_data = download_from_s3(ticker)

        # Si des données existent, on prend la dernière date, sinon on prend 6 mois de données
        if not existing_data.empty:
            last_date = pd.to_datetime(existing_data['Date'].max())
            start_date = last_date + timedelta(days=1)
        else:
            start_date = datetime.now() - timedelta(days=6*30)  # 6 mois en arrière

        # Télécharger les nouvelles données depuis Yahoo Finance
        new_data = download_new_stock_data(ticker, start_date=start_date.strftime('%Y-%m-%d'))

        if not new_data.empty:
            updated_data = pd.concat([existing_data, new_data]).drop_duplicates()
            upload_to_s3(updated_data, ticker)
            print(f"Data for {ticker} updated and uploaded to S3.")
        else:
            print(f"No new data for {ticker}.")


# Handler Lambda
def lambda_handler(event, context):
    """Le point d'entrée principal pour AWS Lambda"""
    update_stock_data()
