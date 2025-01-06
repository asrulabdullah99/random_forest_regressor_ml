import streamlit as st 
import numpy as np 
import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

st.title('Aplikasi Machine Learning')
st.info('Aplikasi ini dibuat menggunakan model Support Vector Regression dan Random Forest Regresor')

with st.expander('Data'):
    st.write('Data Mentah')
    data = pd.read_csv('FuelConsumptionCo2.csv')
    data

with st.expander('Exploratory Data Analysis'):
    st.write(f'Ukuran dataset : {data.shape}')
    
    st.success('Informasi Feature')
    data1 = pd.DataFrame(data)
    buffer = io.StringIO()
    data1.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    
    st.success('Analisa Univariat')
    stats = data.describe()
    stats
    
    st.success('Statistik Setiap Fitur')
    st.write('Rata-Rata Fuel Consumption City : {0: .2f}'.format(data['FUELCONSUMPTION_CITY'].mean()))
    st.write('Rata-Rata Fuel Consumption Highway : {0: .2f}'.format(data['FUELCONSUMPTION_HWY'].mean()))
    st.write('Rata-Rata Engine Size : {0: .2f}'.format(data['ENGINESIZE'].mean()))

with st.expander('Distribusi / Plotting'):
    st.info('Distribusi Sebaran Produsen Mobil')
    fig, ax = plt.subplots()
    sns.histplot(data['MAKE'], color='lightgreen')
    plt.xlabel('Produsen', fontsize=12)
    plt.xticks(rotation='vertical')
    st.pyplot(fig)
    
    st.info('Distribusi Sebaran Tipe Kendaraan')
    fig, ax = plt.subplots()
    sns.histplot(data['VEHICLECLASS'], color='red')
    plt.xlabel('Jenis Kendaraan', fontsize=12)
    plt.xticks(rotation='vertical')
    st.pyplot(fig)
    
with st.expander('Data Preprocessing'):
    st.info('Korelasi antar Fitur Numerical')
    numeric_features = ['MODELYEAR','ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','FUELCONSUMPTION_COMB_MPG','CO2EMISSIONS']
    matriks_korelasi = data[numeric_features].corr()
    
    fig, ax = plt.subplots()
    sns.heatmap(matriks_korelasi,annot=True, cmap='RdBu',annot_kws={"size":8})
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.title('Heatmap Correlation', fontsize=15)
    st.pyplot(fig)

    st.success('Box Plot dan Outlier Detection dengan IQR methods')
    def plot_outliers(data, column):
        st.set_option('deprecation.showPyplotGlobalUse', False)
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        sns.boxplot(data[column])
        plt.title(f'{column} - Box Plot')
        
        plt.subplot(1,2,2)
        sns.histplot(data[column])
        plt.title(f'{column} - Box Plot')
    
    st.pyplot(plot_outliers(data,'ENGINESIZE'))
    st.pyplot(plot_outliers(data,'CYLINDERS'))
    st.pyplot(plot_outliers(data,'FUELCONSUMPTION_CITY'))
    st.pyplot(plot_outliers(data,'FUELCONSUMPTION_HWY'))
    st.pyplot(plot_outliers(data,'FUELCONSUMPTION_COMB'))
    st.pyplot(plot_outliers(data,'FUELCONSUMPTION_COMB_MPG'))
    
    def remove_outliers(data, column):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3-Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR 
        data = data[(data[column] >= lower) &(data[column] <= upper)]
        return data 
    
    data = remove_outliers(data, 'ENGINESIZE')
    data = remove_outliers(data, 'FUELCONSUMPTION_CITY')
    data = remove_outliers(data, 'FUELCONSUMPTION_HWY')
    data = remove_outliers(data, 'FUELCONSUMPTION_COMB')
    data = remove_outliers(data, 'FUELCONSUMPTION_COMB_MPG')

    st.success('After Removed Outlier')
    st.pyplot(plot_outliers(data,'ENGINESIZE'))
    st.pyplot(plot_outliers(data,'CYLINDERS'))
    st.pyplot(plot_outliers(data,'FUELCONSUMPTION_CITY'))
    st.pyplot(plot_outliers(data,'FUELCONSUMPTION_HWY'))
    st.pyplot(plot_outliers(data,'FUELCONSUMPTION_COMB'))
    st.pyplot(plot_outliers(data,'FUELCONSUMPTION_COMB_MPG'))
    
    st.success('Jumlah data hasil remove outlier')
    st.write(f'Ukuran dataset : {data.shape}')
    
with st.expander('Data Partition'):
    st.write('Data Splitting')
    data=data.drop(['MODELYEAR','MAKE','MODEL','VEHICLECLASS','TRANSMISSION','FUELTYPE'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(data.drop(['CO2EMISSIONS'],axis=1), 
                                                        data['CO2EMISSIONS'], 
                                                        test_size=0.30, 
                                                        random_state=42)
    st.info('Data Training')
    st.write(X_train.shape)

with st.expander('Model'):
    st.success('Train Random Forest')
    rfr_model = RandomForestRegressor(max_depth=2, random_state=0)
    rfr_model.fit(X_train, y_train)
    
    y_pred_rfr = rfr_model.predict(X_test)
    st.write("Random Forest Regressor")
    st.write("MAPE", mean_absolute_percentage_error(y_test, y_pred_rfr))
    st.write("MAE", mean_absolute_error(y_test, y_pred_rfr))

with st.expander('Prediksi CO2 Emission'):
    def prediksi (ENGINESIZE,CYLINDERS,FUELCONSUMPTION_CITY,FUELCONSUMPTION_HWY,FUELCONSUMPTION_COMB,FUELCONSUMPTION_COMB_MPG):
        features = np.array([[ENGINESIZE,CYLINDERS,FUELCONSUMPTION_CITY,FUELCONSUMPTION_HWY,FUELCONSUMPTION_COMB,FUELCONSUMPTION_COMB_MPG]])
        prediksi = rfr_model.predict(features).reshape(1,-1)
        
        return prediksi[0]
    
    
    predict = prediksi(3,20,25,23,20,14)
    
    st.write(predict)                    