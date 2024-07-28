import pandas as pd
import iisignature
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score
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
y_array = y.reshape(-1, 1)
Y_scaled = scaler.fit_transform(y_array)

Y_scaled = Y_scaled.flatten()


# Simulation parameters
simulations = 30
rmse_list_train=[]
rmse_list_test=[]
mape_list_train=[]
mean_values = []
mape_list_test=[]
rmspe_list_train=[]
rmspe_list_test=[]
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
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2,random_state=30+_)
    # Fit the model
    model = Lasso(alpha=0.001)
    model.fit(X_train, y_train)
    # Predictions
    predictions = model.predict(X_test)
    # Metrics
    pre_train=model.predict(X_train)

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
    predicted_original_test = np.array(predicted_original_test).flatten()
    y_test_ori = np.array(y_test_ori).flatten()
    diff_percentage = np.abs((predicted_original_test - y_test_ori) / y_test_ori) * 100

    df = pd.DataFrame({
        'y_test_ori': y_test_ori,
        'predicted_original_test': predicted_original_test,
        'diff_percentage': diff_percentage
    })

# top 10
    top_10_diff = df.sort_values(by='diff_percentage', ascending=False).head(10)

# average
    average_top_10_diff = top_10_diff['diff_percentage'].mean()
    print(f"mean:{average_top_10_diff:.3f}")

    mean_values.append(average_top_10_diff)

# Compute average metrics
overall_mean = sum(mean_values) / len(mean_values)
print(f" overall mean: {overall_mean:.3f}")


# Compute average metrics
avg_rmse_train = np.mean(rmse_list_train)
avg_rmse_test = np.mean(rmse_list_test)

avg_mape_train = np.mean(mape_list_train)
avg_mape_test = np.mean(mape_list_test)

avg_rmspe_train = np.mean(rmspe_list_train)
avg_rmspe_test = np.mean(rmspe_list_test)

# Output average results
print(f"Train RMSE(LA): {avg_rmse_train:.3f}")
print(f"Test RMSE(LA): {avg_rmse_test:.3f}")
print(f"Train MAPE(LA): {avg_mape_train:.3f}%")
print(f"Test MAPE(LA): {avg_mape_test:.3f}%")
print(f"Train RMSPE(LA): {avg_rmspe_train:.3f}%")
print(f"Test RMSPE(LA): {avg_rmspe_test:.3f}%")

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
