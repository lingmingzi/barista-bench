import numpy as np
import pandas as pd


def load_csv(path):
    df = pd.read_csv(path)
    configs = {}
    for _, row in df.iterrows():
        id_str = row['id']
        n = int(id_str[:3])
        idx = int(id_str.split('_')[1])
        x_val = str(row['x']).lstrip('s')
        y_val = str(row['y']).lstrip('s')
        deg_val = str(row['deg']).lstrip('s')
        x = float(x_val) if x_val else 0.0
        y = float(y_val) if y_val else 0.0
        deg = float(deg_val) if deg_val else 0.0
        if n not in configs:
            configs[n] = {'x': np.zeros(200), 'y': np.zeros(200), 'deg': np.zeros(200)}
        configs[n]['x'][idx] = x
        configs[n]['y'][idx] = y
        configs[n]['deg'][idx] = deg
    return configs


def save_csv(path, configs):
    rows = []
    for n in sorted(configs.keys()):
        config = configs[n]
        for i in range(n):
            rows.append({
                'id': f'{n:03d}_{i}',
                'x': f's{config["x"][i]:.10f}',
                'y': f's{config["y"][i]:.10f}',
                'deg': f's{config["deg"][i]:.10f}'
            })
    pd.DataFrame(rows).to_csv(path, index=False)
