import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import expon, reciprocal
import iisignature

# Read and preprocess the data
file_path = 'nvda-6.csv'
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
X = scaler.fit_transform(signature_array)
X_scaled = X[20:-1]
X_scaled_1=closing_prices[20:-1].reshape(-1,1)
X_scaled_1 = scaler.fit_transform(X_scaled_1)
y = closing_prices[21:]
y_scaler = StandardScaler()
y_array = y.reshape(-1, 1)
Y_scaled = y_scaler.fit_transform(y_array)

Y_scaled = Y_scaled.flatten()
# Simulation parameters
simulations = 30
rmse_list_train=[]
rmse_list_test=[]
mape_list_train=[]
mape_list_test=[]
rmspe_list_train=[]
rmspe_list_test=[]
rmse_list_new=[]
mape_list_new=[]
rmspe_list_new=[]

param_distributions = {
    'C': reciprocal(1e-4, 1e4),
    'gamma': expon(scale=1.0),
    'epsilon': reciprocal(1e-4, 1e-1)
}
# Function to calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero_indices = y_true != 0
    return np.mean(np.abs((y_true[nonzero_indices] - y_pred[nonzero_indices]) / y_true[nonzero_indices])) * 100

# Function to calculate RMSPE
def root_mean_square_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero_indices = y_true != 0
    return np.sqrt(np.mean(((y_true[nonzero_indices] - y_pred[nonzero_indices]) / y_true[nonzero_indices]) ** 2)) * 100

for _ in range(simulations):
    # Split the data without specifying random_state for variability
    X_train, X_test, y_train, y_test = train_test_split(X_scaled_1, Y_scaled, test_size=0.2,random_state=30+_)
    # Fit the model
    svr = SVR(kernel='rbf')
    random_search = RandomizedSearchCV(svr, param_distributions, n_iter=10, cv=5, verbose=0, random_state=42, n_jobs=-1)
    random_search.fit(X_train, y_train)

    # Best model from random search
    #model = random_search.best_estimator_
    model = SVR(kernel='rbf',C=450,epsilon=0.00043,gamma=0.2)
    model.fit(X_train, y_train)
    # Predictions
    predictions = model.predict(X_test)
    # Metrics
    pre_train=model.predict(X_train)



    pre_train_array= pre_train.reshape(-1,1)
    predictions_array = predictions.reshape(-1,1)
    predicted_original_train = y_scaler.inverse_transform(pre_train_array)
    predicted_original_test = y_scaler.inverse_transform(predictions_array)
    y_test_array = y_test.reshape(-1,1)
    y_train_array=y_train.reshape(-1,1)
    y_test_ori = y_scaler.inverse_transform(y_test_array)
    y_train_ori = y_scaler.inverse_transform(y_train_array)

    rmse_train=np.sqrt(mean_squared_error(y_train_ori,predicted_original_train))
    rmse_test=np.sqrt(mean_squared_error(y_test_ori,predicted_original_test))
    mape_train= mean_absolute_percentage_error(y_train_ori,predicted_original_train)
    mape_test=mean_absolute_percentage_error(y_test_ori,predicted_original_test)
    rmspe_train=root_mean_square_percentage_error(y_train_ori,predicted_original_train)
    rmspe_test=root_mean_square_percentage_error(y_test_ori,predicted_original_test)
    # Collect 

    rmse_list_train.append(rmse_train)
    rmse_list_test.append(rmse_test)

    mape_list_train.append(mape_train)
    mape_list_test.append(mape_test)

    rmspe_list_train.append(rmspe_train)
    rmspe_list_test.append(rmspe_test)


# Compute average metrics
avg_rmse_new=np.mean(rmse_list_new)
avg_mape_new=np.mean(mape_list_new)
avg_rmspe_new=np.mean(rmspe_list_new)
avg_rmse_train = np.mean(rmse_list_train)
avg_rmse_test = np.mean(rmse_list_test)

avg_mape_train = np.mean(mape_list_train)
avg_mape_test = np.mean(mape_list_test)

avg_rmspe_train = np.mean(rmspe_list_train)
avg_rmspe_test = np.mean(rmspe_list_test)

# Output average results
print(f"Train RMSE(SVR): {avg_rmse_train:.3f}")
print(f"Test RMSE(SVR): {avg_rmse_test:.3f}")
print(f"Train MAPE(SVR): {avg_mape_train:.3f}%")
print(f"Test MAPE(SVR): {avg_mape_test:.3f}%")
print(f"Train RMSPE(SVR): {avg_rmspe_train:.3f}%")
print(f"Test RMSPE(SVR): {avg_rmspe_test:.3f}%")
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

