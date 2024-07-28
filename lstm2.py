import numpy as np
import pandas as pd
import iisignature
from keras.wrappers.scikit_learn import KerasRegressor
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping # type: ignore

def create_model(units=50, learning_rate=0.01):
    model = Sequential()
    model.add(LSTM(units, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
    model.add(Dense(1, activation='linear'))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    return model

# Read and preprocess the data
file_path = 'nvda-4.csv'
data = pd.read_csv(file_path)
data = data.iloc[:, :-1]
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values('Date', inplace=True)
data.dropna(inplace=True)

dates = data['Date'].values
closing_prices = data['Close'].values
path = list(enumerate(closing_prices))
order = 4
signatures = [iisignature.sig(path[:i+1], order) for i in range(len(path))]
signature_array = np.array(signatures)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(signature_array)
y = closing_prices
y_array = y.reshape(-1, 1)
Y_scaled = scaler.fit_transform(y_array)

Y_scaled = Y_scaled.flatten()
# Function to create dataset for training/testing
def create_dataset(X_scaled, Y_scaled,look_back=10):
    dataX, dataY = [], []
    for i in range(len(Y_scaled) - look_back):
        a = Y_scaled[i:(i + look_back)]
        dataX.append(a)
        dataY.append(Y_scaled[i + look_back])
    return np.array(dataX), np.array(dataY)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero_indices = y_true !=0
    return np.mean(np.abs((y_true[nonzero_indices] - y_pred[nonzero_indices]) / y_true[nonzero_indices])) * 100
def root_mean_square_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero_indices = y_true !=0 
    return np.sqrt(np.mean(((y_true[nonzero_indices] - y_pred[nonzero_indices]) / y_true[nonzero_indices]) ** 2)) * 100

X, y = create_dataset(X_scaled,Y_scaled, 20)
simulations = 30
rmse_list_train=[]
rmse_list_test=[]
mape_list_train=[]
mape_list_test=[]
rmspe_list_train=[]
rmspe_list_test=[]

param_grid = {
    'units': [50, 100, 150],
    'learning_rate': [0.1,0.01, 0.001, 0.0001]
}
for i in range(simulations):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30+i)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
# Build CNN model

    model = KerasRegressor(build_fn=create_model, verbose=0)

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search_result = grid_search.fit(X_train, y_train)

    best_model = grid_search_result.best_estimator_


    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


    history = best_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping])


    predictions = best_model.predict(X_test)
    pre_train = best_model.predict(X_train)
    pre_train_array= pre_train.reshape(-1,1)
    predictions_array = predictions.reshape(-1,1)
    predicted_original_train = scaler.inverse_transform(pre_train_array)
    predicted_original_test = scaler.inverse_transform(predictions_array)
    y_test_array = y_test.reshape(-1,1)
    y_train_array=y_train.reshape(-1,1)
    y_test_ori = scaler.inverse_transform(y_test_array)
    y_train_ori = scaler.inverse_transform(y_train_array)
 
    rmse_train=np.sqrt(mean_squared_error(y_train_ori,predicted_original_train))
    rmse_test=np.sqrt(mean_squared_error(y_test_ori,predicted_original_test))

    mape_train = mean_absolute_percentage_error(y_train_ori,predicted_original_train)
    mape_test = mean_absolute_percentage_error(y_test_ori,predicted_original_test)
    rmspe_train = root_mean_square_percentage_error(y_train_ori,predicted_original_train)
    rmspe_test = root_mean_square_percentage_error(y_test_ori,predicted_original_test)

    # Collect 
    rmse_list_train.append(rmse_train)
    rmse_list_test.append(rmse_test)

    mape_list_train.append(mape_train)
    mape_list_test.append(mape_test)

    rmspe_list_train.append(rmspe_train)
    rmspe_list_test.append(rmspe_test)


avg_rmse_train = np.mean(rmse_list_train)
avg_rmse_test = np.mean(rmse_list_test)

avg_mape_train = np.mean(mape_list_train)
avg_mape_test = np.mean(mape_list_test)

avg_rmspe_train = np.mean(rmspe_list_train)
avg_rmspe_test = np.mean(rmspe_list_test)

# Output average results
print(f"Train RMSE(LSTM): {avg_rmse_train:.3f}")
print(f"Test RMSE(LSTM): {avg_rmse_test:.3f}")
print(f"Train MAPE(LSTM): {avg_mape_train:.3f}%")
print(f"Test MAPE(LSTM): {avg_mape_test:.3f}%")
print(f"Train RMSPE(LSTM): {avg_rmspe_train:.3f}%")
print(f"Test RMSPE(LSTM): {avg_rmspe_test:.3f}%")
std_rmse_test = np.std(rmse_list_test, ddof=1)
std_rmse_train = np.std(rmse_list_train, ddof=1)
std_mape_test = np.std(rmspe_list_test, ddof=1)
std_mape_train = np.std(rmspe_list_train, ddof=1)
std_rmspe_train = np.std(rmspe_list_train)
std_rmspe_test = np.std(rmspe_list_test)
# Sample sizes
n_rmse_test = len(rmse_list_test)
n_rmse_train = len(rmse_list_train)
n_mape_test = len(rmspe_list_test)
n_mape_train = len(rmspe_list_train)
n_rmspe_train = len(rmspe_list_train)
n_rmspe_test = len(rmspe_list_test)
# Z-score for 95% confidence
z = norm.ppf(0.975)  

# Confidence intervals
ci_rmse_test = (avg_rmse_test - z * (std_rmse_test / np.sqrt(n_rmse_test)),
                avg_rmse_test + z * (std_rmse_test / np.sqrt(n_rmse_test)))
ci_rmse_train = (avg_rmse_train - z * (std_rmse_train / np.sqrt(n_rmse_train)),
           avg_rmse_train + z * (std_rmse_train / np.sqrt(n_rmse_train)))
ci_mape_test = (avg_mape_test - z * (std_mape_test / np.sqrt(n_mape_test)),
                avg_mape_test + z * (std_mape_test / np.sqrt(n_mape_test)))
ci_mape_train = (avg_mape_train - z * (std_mape_train / np.sqrt(n_mape_train)),
           avg_mape_train + z * (std_mape_train / np.sqrt(n_mape_train)))
ci_rmspe_test = (avg_rmspe_test - z * (std_rmspe_test / np.sqrt(n_rmspe_test)),
                avg_rmspe_test + z * (std_rmspe_test / np.sqrt(n_rmspe_test)))
ci_rmspe_train = (avg_rmspe_train - z * (std_rmspe_train / np.sqrt(n_rmspe_train)),
           avg_rmspe_train + z * (std_rmspe_train / np.sqrt(n_rmspe_train)))

print("95% CI for RMSE (test):", ci_rmse_test)
print("95% CI for RMSE (Train):", ci_rmse_train)
print("95% CI for MAPE (test):", ci_mape_test)
print("95% CI for MAPE (Train):", ci_mape_train)
print("95% CI for RMSPE (test):", ci_rmspe_test)
print("95% CI for RMSPE (Train):", ci_rmspe_train)

