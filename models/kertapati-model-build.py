import os
import pyodbc
import pandas as pd
from dotenv import load_dotenv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner import RandomSearch
from sklearn.preprocessing import MinMaxScaler

load_dotenv()

try:
    server = os.environ['SERVER']
    database = os.environ['DATABASE']
    username = os.environ['USERNAME_1']
    password = os.environ['PASSWORD']

    cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+password)
    cursor = cnxn.cursor()
    query = "SELECT * FROM dwh_prod.dwh.DM_transport_transaction;" 

    def process_transport_data(cnxn, query):
        try:
            data = pd.read_sql(query, cnxn)
            df_transport_real = data[data['jenis'] == 'Real']
            df_pivot = df_transport_real.pivot_table(
                index=['tanggal', 'jenis'], 
                columns='dest_name', 
                values='tonase', 
                aggfunc='sum'
            ).reset_index()
            df_pivot['total'] = df_pivot.sum(axis=1, numeric_only=True)
            df_pivot['tanggal'] = pd.to_datetime(df_pivot['tanggal'], format='%Y-%m-%d')
            blueprint = df_pivot[['tanggal', 'Kertapati', 'Tarahan', 'total']].copy()
            df_kertapati = blueprint.copy()
            df_kertapati['Kertapati'].fillna(df_kertapati['Kertapati'].median(), inplace=True)
            df_kertapati['tanggal'] = pd.to_datetime(df_kertapati['tanggal'])
            df_kertapati = df_kertapati.sort_values('tanggal').set_index('tanggal')

            return df_kertapati

        except Exception as e:
            print("Terjadi kesalahan saat memproses data:", e)
            return None

    processed_data = process_transport_data(cnxn, query)
    data_k = processed_data['Kertapati'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data_k_scaled = scaler.fit_transform(data_k)

    def create_sequences(data_k, seq_length):
        xs = []
        ys = []
        for i in range(len(data_k) - seq_length):
            x = data_k[i:(i + seq_length)]
            y = data_k[i + seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)
    
    seq_length = 30
    X_k, y_k = create_sequences(data_k_scaled, seq_length)
    train_size_k = int(len(X_k) * 0.8)
    X_train_k, X_test_k = X_k[:train_size_k], X_k[train_size_k:]
    y_train_k, y_test_k = y_k[:train_size_k], y_k[train_size_k:]

    # Model Build
    def build_model(hp):
        model = Sequential()
        model.add(LSTM(hp.Int('units_1', min_value=50, max_value=80, step=10),
                    activation='relu',
                    input_shape=(seq_length, 1),
                    return_sequences=True))
        model.add(LSTM(hp.Int('units_2', min_value=50, max_value=80, step=10),
                    activation='relu'))
        model.add(Dense(1))

        model.compile(optimizer=Adam(learning_rate=hp.Float('learning_rate', 1e-4, 1e-3, sampling='LOG')),
                    loss='mse', metrics=['mae'])
        return model

    tuner = RandomSearch(
        build_model,
        objective='val_mae',
        max_trials=10,
        executions_per_trial=1,
        directory='my_dir',
        project_name='kertapati_tuner'
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    try:
        tuner.search(X_train_k, y_train_k,
                    epochs=100,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    verbose=1)
    except Exception as e:
        print(f"Error during tuning: {e}")

    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Best Hyperparameters: {best_hyperparameters.values}")

    model_k = tuner.hypermodel.build(best_hyperparameters)
    history_k = model_k.fit(X_train_k, y_train_k,
                        epochs=100,
                        batch_size=32,
                        validation_split=0.2,
                        callbacks=[early_stopping],
                        verbose=1)

    model_k.save('kertapati_lstm_model.h5')


except Exception as e:
    print("Terjadi kesalahan dalam koneksi atau eksekusi query:", e)
finally:
    if 'cnxn' in locals():
        cnxn.close()