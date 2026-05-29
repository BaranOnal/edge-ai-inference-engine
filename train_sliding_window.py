import numpy as np
import pandas as pd
import json

COLS = (['unit','cycle','op1','op2','op3'] +
        [f's{i}' for i in range(1, 22)])

# Train data
df_train = pd.read_csv('archive/train_FD001.txt', sep=r'\s+', header=None, names=COLS)

# RUL->Remaining Useful Life
max_cycle = df_train.groupby('unit')['cycle'].max().reset_index()
max_cycle.columns = ['unit', 'max_cycle']
df_train = df_train.merge(max_cycle, on='unit')
df_train['rul']   = df_train['max_cycle'] - df_train['cycle']
# # 30 cycles <= -> anomaly 1, others normal 0
df_train['label'] = (df_train['rul'] <= 30).astype(int)

# Test data
# test_FD001.txt: each engine is cut off at its last moment, rul unknown
# RUL_FD001.txt: true remaining useful life of each engine
df_test = pd.read_csv('archive/test_FD001.txt', sep=r'\s+', header=None, names=COLS)
rul_test = pd.read_csv('archive/RUL_FD001.txt',  sep=r'\s+', header=None, names=['rul'])

# sensor data
SENSORS   = ['s2', 's3', 's4', 's7']
N_SENSORS = len(SENSORS)

# train normalizastion
feat_train = df_train[SENSORS].values.astype(np.float32)
feat_min   = feat_train.min(axis=0)
feat_max   = feat_train.max(axis=0)
feat_train = (feat_train - feat_min) / (feat_max - feat_min + 1e-8)
labels_train = df_train['label'].values.astype(np.float32)
units_train  = df_train['unit'].values

# sliding window for train
WINDOW = 50
X_list, y_list = [], []
for unit_id in df_train['unit'].unique():
    mask   = units_train == unit_id
    f_unit = feat_train[mask]
    l_unit = labels_train[mask]#different engines should not be in the same window
    for i in range(WINDOW, len(f_unit)):
        X_list.append(f_unit[i-WINDOW:i].flatten())
        y_list.append(l_unit[i])

X_train = np.array(X_list, dtype=np.float32).T #shape-> (200, N)
y_train = np.array(y_list, dtype=np.float32).reshape(1, -1)

print(f"Train set: {X_train.shape}, Anomaly Ratio: {y_train.mean():.2%}")

# test normalization
feat_test_raw = df_test[SENSORS].values.astype(np.float32)
feat_test_raw = (feat_test_raw - feat_min) / (feat_max - feat_min + 1e-8)
units_test    = df_test['unit'].values

# sliding window for test
X_test_list, y_test_list = [], []
for unit_id in df_test['unit'].unique():
    mask   = units_test == unit_id
    f_unit = feat_test_raw[mask]
    # last 50 cycles, if less than 50 cycles, pad with zeros at the beginning
    if len(f_unit) >= WINDOW:
        window = f_unit[-WINDOW:].flatten()
    else:
        pad = np.zeros((WINDOW - len(f_unit), N_SENSORS), dtype=np.float32)
        window = np.vstack([pad, f_unit]).flatten()
    X_test_list.append(window)
    # True label: is the RUL of that engine <= 30
    real_rul = rul_test.iloc[unit_id - 1]['rul']
    y_test_list.append(int(real_rul <= 30))

X_test = np.array(X_test_list, dtype=np.float32).T   # (200, 100)
y_test = np.array(y_test_list, dtype=np.float32).reshape(1, -1)

print(f"Test set: {X_test.shape}, Anomaly Ratio: {y_test.mean():.2%}")

# Model  
import nn.layers as nn_framework

layers_dims = [200, 128, 64, 1]

model = nn_framework.NeuralNetwork(
    X_train, y_train, layers_dims,
    activation_function="relu",
    loss_function="binary_crossentropy",
    learning_rate=5e-4,
    optimizer="Adam",
    initialization="he",
    validation_split=0.0
)

model.fit(epochs=30, batch_size=256)


def predict(model, X):
    """use the model's forward pass."""
    return model.forward(X)


probs = predict(model, X_test)   # (1, 100)
preds = (probs >= 0.5).astype(int).flatten()
truth = y_test.flatten().astype(int)

TP = int(((preds == 1) & (truth == 1)).sum())
TN = int(((preds == 0) & (truth == 0)).sum())
FP = int(((preds == 1) & (truth == 0)).sum())
FN = int(((preds == 0) & (truth == 1)).sum())

accuracy  = (TP + TN) / len(truth)
precision = TP / (TP + FP + 1e-8)
recall    = TP / (TP + FN + 1e-8)
f1        = 2 * precision * recall / (precision + recall + 1e-8)

print("\n========== TEST RESULTS ==========")
print(f"Accuracy : {accuracy:.3f}  ({accuracy*100:.1f}%)")
print(f"Precision: {precision:.3f}")
print(f"Recall   : {recall:.3f}")
print(f"F1 Score : {f1:.3f}")
print(f"\nConfusion Matrix:")
print(f"  TP={TP}  FP={FP}")
print(f"  FN={FN}  TN={TN}")
print(f"\nNumber of test samples: {len(truth)}")
print(f"True Anomalies    : {truth.sum()}")
print(f"Predicted Anomalies    : {preds.sum()}")
print("=====================================\n")

#Exporting weights
params    = model.get_parameters
n_layers = sum(1 for k in model.get_parameters if k.startswith('W'))

export = {
    "architecture": layers_dims,
    "n_sensors"   : N_SENSORS,
    "window_size" : WINDOW,
    "feat_min"    : feat_min.tolist(),
    "feat_max"    : feat_max.tolist(),
    "layers"      : []
}

for i in range(1, n_layers + 1):
    W = params.get(f'W{i}', np.array([]))
    b = params.get(f'b{i}', np.array([])).flatten()
    export["layers"].append({
        "W"         : W.tolist(),
        "b"         : b.tolist(),
        "activation": "relu" if i < n_layers else "sigmoid"
    })

with open("weights.json", "w") as f:
    json.dump(export, f)

print("weights.json saved.")


import time

# Measure on a single sample (same condition as C)
X_single = X_test[:, :1]   # (200, 1)

# Warm-up (eliminate JIT effect)
for _ in range(100):
    model.forward(X_single)

# Benchmark — 10000 iterations
N = 10000   
t0 = time.perf_counter()
for _ in range(N):
    model.forward(X_single)
t1 = time.perf_counter()

elapsed    = t1 - t0
per_sample = elapsed / N * 1000   # ms

print(f"\n===== PYTHON FORWARD PASS =====")
print(f"Time         : {elapsed:.4f} s")
print(f"Throughput   : {N/elapsed:.0f} packet/s")
print(f"Avg. latency : {per_sample:.4f} ms/packet")
print(f"=====================================")