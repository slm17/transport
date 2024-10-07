import os
import pyodbc
import pandas as pd
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
from keras.losses import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np

load_dotenv()

try:
    server = os.environ['SERVER']
    database = os.environ['DATABASE']
    username = os.environ['USERNAME_1']
    password = os.environ['PASSWORD']


    cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+password)
    cursor = cnxn.cursor()

    query = "SELECT * FROM dwh_prod.dwh.DM_transport_transaction;" 

    data = pd.read_sql(query, cnxn)
    df_transport_real = data[data['jenis'] == 'Real']
    df_pivot = df_transport_real.pivot_table(index=['tanggal', 'jenis'], columns='dest_name', values='tonase', aggfunc='sum').reset_index()
    df_pivot['total'] = df_pivot.sum(axis=1, numeric_only=True)
    df_pivot['tanggal'] = pd.to_datetime(df_pivot['tanggal'], format='%Y-%m-%d')

    blueprint = df_pivot[['tanggal', 'Kertapati', 'Tarahan', 'total']].copy()
    df = blueprint.copy()
    df['Kertapati'].fillna(df['Kertapati'].median(), inplace=True)
    df['Tarahan'].fillna(df['Tarahan'].median(), inplace=True)
    df['total'].fillna(df['total'].median(), inplace=True)
    df['tanggal'] = pd.to_datetime(df['tanggal'])
    df = df.sort_values('tanggal').set_index('tanggal')
    
    # load models
    # kertapati_model = load_model('./models/kertapati_lstm_model.h5', custom_objects={'mse': mean_squared_error})
    # tarahan_model = load_model('./models/tarahan_lstm_model.h5', custom_objects={'mse': mean_squared_error})
    # total_model = load_model('./models/total_lstm_model.h5', custom_objects={'mse': mean_squared_error})

    def predict_column_future(df, column_name, model, look_back=30, num_future_days=150):
        train_data = df[[column_name]].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_data_scaled = scaler.fit_transform(train_data)

        X_test = []
        for i in range(look_back, len(train_data_scaled)):
            X_test.append(train_data_scaled[i-look_back:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        predicted_scaled = model.predict(X_test)

        predicted_column = scaler.inverse_transform(predicted_scaled)

        last_30_days = train_data_scaled[-look_back:]
        predicted_future = []
        for _ in range(num_future_days):
            X_input = np.reshape(last_30_days, (1, look_back, 1))
            predicted_scaled_future = model.predict(X_input)
            predicted_future.append(predicted_scaled_future[0][0])
            last_30_days = np.append(last_30_days[1:], predicted_scaled_future[0])

        predicted_future = scaler.inverse_transform(np.array(predicted_future).reshape(-1, 1))

        combined_data = np.concatenate((train_data, predicted_column, predicted_future))

        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(days=1), periods=num_future_days)

        future_df = pd.DataFrame({column_name: predicted_future.flatten()}, index=future_dates)

        combined_df = pd.concat([df[column_name], future_df])

        return predicted_future, combined_df

    # Load models
    kertapati_model = load_model('./models/kertapati_lstm_model.h5', custom_objects={'mse': mean_squared_error})
    tarahan_model = load_model('./models/tarahan_lstm_model.h5', custom_objects={'mse': mean_squared_error})
    total_model = load_model('./models/total_lstm_model.h5', custom_objects={'mse': mean_squared_error})

    # Apply the function for each column
    predicted_kertapati, combined_kertapati = predict_column_future(df, 'Kertapati', kertapati_model)
    predicted_tarahan, combined_tarahan = predict_column_future(df, 'Tarahan', tarahan_model)
    predicted_total, combined_total = predict_column_future(df, 'total', total_model)

    # Combine all predictions into one DataFrame
    combined_df = pd.concat([combined_kertapati, combined_tarahan, combined_total], axis=1)

    # Create a copy of the original DataFrame
    forecast_df = data.copy()

    # Create a new DataFrame for the forecast data
    forecast_data = combined_df.reset_index()
    forecast_data = forecast_data.rename(columns={'index': 'tanggal'})
    forecast_data['jenis'] = 'Forecast'

    # Melt the forecast data to match the original DataFrame structure
    forecast_data_melted = pd.melt(forecast_data, id_vars=['tanggal', 'jenis'], var_name='dest_name', value_name='tonase')

    # Append the forecast data to the original DataFrame
    forecast_df = pd.concat([forecast_df, forecast_data_melted], ignore_index=True)

    forecast_df_selected = forecast_df[['tanggal', 'jenis', 'dest_name', 'tonase']]

    dd = forecast_df_selected[['tanggal', 'jenis', 'dest_name', 'tonase']].astype(str)
    dd['tanggal'] = pd.to_datetime(dd['tanggal'])

    # Insert into DB
    for index, row in dd.iterrows():
        cursor.execute("""
            IF EXISTS (SELECT 1 FROM DM_forecasting_transport_transaction_d WHERE tanggal = ?)
            BEGIN
                UPDATE DM_forecasting_transport_transaction_d
                SET jenis = ?, dest_name = ?, tonase = ?
                WHERE tanggal = ?
            END
            ELSE
            BEGIN
                INSERT INTO DM_forecasting_transport_transaction_d (tanggal, jenis, dest_name, tonase)
                VALUES (?, ?, ?, ?)
            END
            """, 
            index.date(), row['jenis'], row['dest_name'], row['tonase'],
            index.date(),  # Untuk kondisi WHERE di bagian UPDATE
            index.date(), row['jenis'], row['dest_name'], row['tonase'])  # Untuk INSERT
    cnxn.commit()
    print(dd.info())
    print(dd.head())

except Exception as e:
    print("Terjadi kesalahan dalam koneksi atau eksekusi query:", e)
finally:
    if 'cnxn' in locals():
        cnxn.close()