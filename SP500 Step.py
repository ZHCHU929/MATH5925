import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


volatilities = ['T+1', 'T+5','T+10','T+20']
models = ['LSTM', 'Lasso',  'Random Forest', 'SVR']

rmse_sig = {
    'LSTM': [44.559, 66.418, 65.726, 66.340],
    'Lasso': [43.684, 87.923, 114.691, 137.613],
    'Random Forest': [49.080, 62.840, 62.851, 56.922],
    'SVR': [44.115, 69.756, 69.397, 60.460]
}

rmse_non_sig = {
    'LSTM': [44.164, 94.685, 113.328, 130.389],
    'Lasso': [43.772, 91.151, 128.122, 174.829],
    'Random Forest': [52.146, 108.455, 144.639, 188.774],
    'SVR': [49.915, 91.019, 124.852, 165.471]
}

mape_sig = {
    'LSTM': [0.839, 1.249, 1.242, 1.250],
    'Lasso': [0.799, 1.611, 2.127, 2.579],
    'Random Forest': [0.895, 1.077, 1.025, 0.939],
    'SVR': [0.804, 1.228, 1.209, 1.234]
}

mape_non_sig = {
    'LSTM': [0.800, 1.692, 2.116, 2.426],
    'Lasso': [0.801, 1.678, 2.401, 3.363],
    'Random Forest': [0.968, 2.029, 2.681, 3.438],
    'SVR': [0.801, 1.665, 2.296, 2.964]
}

rmspe_sig = {
    'LSTM': [1.113, 1.644, 1.641, 1.665],
    'Lasso': [1.063, 2.152, 2.792, 3.310],
    'Random Forest': [1.192, 1.525, 1.526, 1.366],
    'SVR': [1.071, 1.713, 1.716, 1.747]
}

rmspe_non_sig = {
    'LSTM': [1.077, 2.249, 2.777, 3.141],
    'Lasso': [1.066, 2.237, 3.137, 4.262],
    'Random Forest': [1.263, 2.652, 3.531, 4.584],
    'SVR': [1.068, 2.234, 3.065, 3.997]
}


color_map = {
    'LSTM': ('blue', 'lightblue'),
    'Lasso': ('green', 'lightgreen'),
    'Random Forest': ('red', 'lightcoral'),
    'SVR': ('purple', 'plum')
}
def to_percent(y, position):
    return f'{y:.0f}%'
def plot_metric(metric_sig, metric_non_sig, metric_name):
    bar_width = 0.1
    opacity = 0.8
    index = np.arange(len(volatilities)) * (len(models) * 2 + 1) * bar_width  # Add space between groups
    
    fig, ax = plt.subplots(figsize=(18, 10))
    
    for i, model in enumerate(models):
        if model in metric_sig:
            plt.bar(index + i * 2 * bar_width, metric_sig[model], bar_width,
                    alpha=opacity, color=color_map[model][0], label=f'{model} (Sig)', hatch='\\')
        if model in metric_non_sig:
            plt.bar(index + i * 2 * bar_width + bar_width, metric_non_sig[model], bar_width,
                    alpha=opacity,color=color_map[model][1],  label=f'{model} (Non-Sig)', hatch='/')


    
    plt.xlabel('Time Steps', fontsize=25)
    plt.ylabel(metric_name, fontsize=25)
    plt.title('Comparison Across Using Signature and Non-Signature in Different Steps in S&P 500', fontsize=25)
    plt.xticks(index + bar_width * (len(models) - 1), volatilities, fontsize=25)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.grid(True)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.ylim(0, 5)
    plt.yticks(fontsize=15)
    #plt.show()
    plt.savefig(f'{metric_name}-SP500-T')

# 绘制图表
#plot_metric(rmse_sig, rmse_non_sig, 'RMSE')
plot_metric(mape_sig, mape_non_sig, 'MAPE')
plot_metric(rmspe_sig, rmspe_non_sig, 'RMSPE')
