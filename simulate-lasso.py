import numpy as np
import pandas as pd
import iisignature
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score


def generate_bs_motion(n, S0, dt=1, mu=0.001, sigma=0.01, seed=None):
    if seed is not None:
        np.random.seed(seed)

    W = np.random.standard_normal(size=n)
    W = np.cumsum(W) * np.sqrt(dt)
    t = np.linspace(0, (n-1)*dt, n)

    S = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)
    return S
import numpy as np

def single_monte_carlo_path(S0, T, r, sigma, num_steps, seed=None):

    if seed is not None:
        np.random.seed(seed)
    dt = T / num_steps


    asset_path = np.zeros(num_steps + 1)
    asset_path[0] = S0

    random_numbers = np.random.normal(0, 1, num_steps)

    for t in range(1, num_steps + 1):
        asset_path[t] = asset_path[t - 1] * np.exp(
            (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * random_numbers[t - 1]
        )

    return asset_path


S0 = 1 
T = 4     
r = 0.05  
sigma = 0.1  
num_steps = 1008  


closing_prices_simulated= single_monte_carlo_path(S0, T, r, sigma, num_steps,42)


time_steps = np.linspace(0, T, num_steps + 1)


asset_path_df = pd.DataFrame({
    'Time Step': time_steps,
    'Asset Price': closing_prices_simulated
})


#path = np.column_stack((time_steps, closing_prices_simulated))







# simulated_data.to_csv('simulated_Brownian_motion.csv', index=False)

simulated_data = pd.DataFrame({ 
    'Time': time_steps,
    'Close': closing_prices_simulated
})


print(simulated_data.head())

dates = simulated_data['Time'].values
closing_prices = simulated_data['Close'].values
path = list(enumerate(closing_prices))
order = 4
signatures = [iisignature.sig(path[:i+1], order) for i in range(len(path))]
signature_array = np.array(signatures)
scaler = StandardScaler()
X = scaler.fit_transform(signature_array)
X_scaled = X[20:-20]
X_scaled_1=closing_prices[20:-20].reshape(-1,1)
y = closing_prices[40:]

y_array = y.reshape(-1, 1)
Y_scaled = scaler.fit_transform(y_array)


Y_scaled = Y_scaled.flatten()
# Simulation parameters
simulations = 30
rmse_list_train=[]
rmse_list_test=[]
mape_list_train=[]
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
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2,random_state=30+_)
    # Fit the model
    model = Lasso(alpha=0.001)
    model.fit(X_train, y_train)
    # Predictions
    predictions = model.predict(X_test)
    # Metrics
    pre_train=model.predict(X_train)
    mse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    n = len(y_test)
    p = X_test.shape[1]
    adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    rmse_train=np.sqrt(mean_squared_error(y_train,pre_train))
    rmse_test=np.sqrt(mean_squared_error(y_test,predictions))
    pre_train_array= pre_train.reshape(-1,1)
    predictions_array = predictions.reshape(-1,1)
    predicted_original_train = scaler.inverse_transform(pre_train_array)
    predicted_original_test = scaler.inverse_transform(predictions_array)
    y_test_array = y_test.reshape(-1,1)
    y_train_array=y_train.reshape(-1,1)
    y_test_ori = scaler.inverse_transform(y_test_array)
    y_train_ori = scaler.inverse_transform(y_train_array)
    mape_train= mean_absolute_percentage_error(y_train,pre_train)
    mape_test=mean_absolute_percentage_error(y_test,predictions)
    rmspe_train=root_mean_square_percentage_error(y_train,pre_train)
    rmspe_test=root_mean_square_percentage_error(y_test,predictions)
    # Collect 
    rmse_list_train.append(rmse_train)
    rmse_list_test.append(rmse_test)

    mape_list_train.append(mape_train)
    mape_list_test.append(mape_test)

    rmspe_list_train.append(rmspe_train)
    rmspe_list_test.append(rmspe_test)


# Compute average metrics
avg_rmse_train = np.mean(rmse_list_train)
avg_rmse_test = np.mean(rmse_list_test)

avg_mape_train = np.mean(mape_list_train)
avg_mape_test = np.mean(mape_list_test)

avg_rmspe_train = np.mean(rmspe_list_train)
avg_rmspe_test = np.mean(rmspe_list_test)

# Output average results
print("Train RMSE(LA):", avg_rmse_train)
print("Test RMSE(LA):", avg_rmse_test)
print("Train MAPE(LA):", avg_mape_train,"%")
print("Test MAPE(LA):", avg_mape_test, "%")
print("Train RMSPE(LA):", avg_rmspe_train, "%")  # Print average RMSPE
print("Test RMSPE(LA):",avg_rmspe_test,"%")
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
