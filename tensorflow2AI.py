# %% 패키지 설치
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense

# %% 학습용 데이터(엑셀) 불러오기
x_train = pd.read_excel("dataset\\허브 광량 환경 데이터셋.xlsx", sheet_name="원인").to_numpy().astype(float)
y_train = pd.read_excel("dataset\\허브 광량 환경 데이터셋.xlsx", sheet_name="결과").to_numpy().astype(float)

# %% 검증용 데이터 생성
x_val = np.array([
    [150, 4],    # 낮은 광량, 짧은 시간
    [300, 6],    # 낮은 광량, 짧은 시간
    [500, 8],    # 중간 광량, 중간 시간
    [800, 10],   # 중간 광량, 중간 시간
    [1000, 12],  # 중간-높은 광량, 긴 시간
    [1200, 14],  # 높은 광량, 긴 시간
    [1500, 16],  # 높은 광량, 매우 긴 시간
    [1800, 18],  # 매우 높은 광량, 매우 긴 시간
    [400, 5],    # 낮은 광량, 짧은 시간
    [1100, 11]   # 중간-높은 광량, 중간 시간
], dtype=float)

# 검증용 출력 데이터 (각 식물의 DLI 충족도 %)
# [로즈마리, 스위트바질, 민트, 라벤더, 레몬밤]
y_val = np.array([
    [35, 30, 40, 32, 38],      # 낮은 광량, 짧은 시간
    [48, 42, 55, 45, 50],      # 낮은 광량, 짧은 시간
    [65, 58, 72, 60, 68],      # 중간 광량, 중간 시간
    [85, 78, 95, 80, 88],      # 중간 광량, 중간 시간
    [98, 92, 108, 95, 102],    # 중간-높은 광량, 긴 시간
    [112, 105, 125, 108, 118], # 높은 광량, 긴 시간
    [135, 128, 148, 132, 140], # 높은 광량, 매우 긴 시간
    [158, 150, 172, 155, 165], # 매우 높은 광량, 매우 긴 시간
    [52, 46, 58, 48, 55],      # 낮은 광량, 짧은 시간
    [105, 98, 115, 102, 110]   # 중간-높은 광량, 중간 시간
], dtype=float)

# %% min-max 정규화
# 정규화용 함수 정의
def min_max_scaler(data):
    min_val = np.min(data)
    max_val = np.max(data)
    scaled_data = (data - min_val) / (max_val - min_val)
    return scaled_data, min_val, max_val

def min_max_scaler_input_minmax(data, min_val, max_val):
    scaled_data = (data - min_val) / (max_val - min_val)
    return scaled_data

# 학습용 데이터 정규화
x_train[:,0], x_min_0, x_max_0 = min_max_scaler(x_train[:,0])  # 광량
x_train[:,1], x_min_1, x_max_1 = min_max_scaler(x_train[:,1])  # 광노출시간
y_train[:,0], y_min_0, y_max_0 = min_max_scaler(y_train[:,0])  # 로즈마리 DLI충족도
y_train[:,1], y_min_1, y_max_1 = min_max_scaler(y_train[:,1])  # 스위트바질 DLI충족도
y_train[:,2], y_min_2, y_max_2 = min_max_scaler(y_train[:,2])  # 민트 DLI충족도
y_train[:,3], y_min_3, y_max_3 = min_max_scaler(y_train[:,3])  # 라벤더 DLI충족도
y_train[:,4], y_min_4, y_max_4 = min_max_scaler(y_train[:,4])  # 레몬밤 DLI충족도

# 검증용 데이터 정규화
x_val[:,0] = min_max_scaler_input_minmax(x_val[:,0], x_min_0, x_max_0)  # 광량
x_val[:,1] = min_max_scaler_input_minmax(x_val[:,1], x_min_1, x_max_1)  # 광노출시간
y_val[:,0] = min_max_scaler_input_minmax(y_val[:,0], y_min_0, y_max_0)  # 로즈마리 DLI충족도
y_val[:,1] = min_max_scaler_input_minmax(y_val[:,1], y_min_1, y_max_1)  # 스위트바질 DLI충족도
y_val[:,2] = min_max_scaler_input_minmax(y_val[:,2], y_min_2, y_max_2)  # 민트 DLI충족도
y_val[:,3] = min_max_scaler_input_minmax(y_val[:,3], y_min_3, y_max_3)  # 라벤더 DLI충족도
y_val[:,4] = min_max_scaler_input_minmax(y_val[:,4], y_min_4, y_max_4)  # 레몬밤 DLI충족도

# %% 모델 생성
model = Sequential() 
model.add(Input(shape=(2,)))  # 입력 2개 (광량, 광노출시간)
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(5))  # 출력 5개 (각 허브 DLI충족도)

# %% 모델 학습
optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)
model.compile(optimizer=optimizer, loss='mse')  # mean squared error
hist = model.fit(x_train, y_train, epochs=1000, validation_data=(x_val, y_val))

# %% 예측 수행, 결과 시각화
plant_names = ["Rosemary", "Sweet Basil", "Mint", "Lavender", "Lemon Balm"]

# 그래프 시각화 함수 정의
def plot_vision(ax0, ax1, x_train_0, x_range_0, x_val_0, x_train_1, x_range_1, x_val_1, 
                y_train_i, y_val_i, y_predict_0_i, y_predict_1_i, plant_name):
    # 광량 기준 예측 그래프 시각화
    ax0.scatter(x_train_0, y_train_i, label='Train', c='green')
    ax0.scatter(x_val_0, y_val_i, label='Validation', c='orange')
    ax0.plot(x_range_0, y_predict_0_i, label='Prediction', c='blue')
    ax0.set_xlabel('Light Intensity')
    ax0.set_ylabel(f'{plant_name} DLI Satisfaction')

    # 광노출시간 기준 예측 그래프 시각화
    ax1.scatter(x_train_1, y_train_i, label='Train', c='green')
    ax1.scatter(x_val_1, y_val_i, label='Validation', c='orange')
    ax1.plot(x_range_1, y_predict_1_i, label='Prediction', c='blue')
    ax1.set_xlabel('Light Exposure Time')
    ax1.set_ylabel(f'{plant_name} DLI Satisfaction')

# %% 기본 버전: 기준 이외 원인 고정 X, 역정규화 X
fig = plt.figure(figsize=(20, 12))
for i in range(5):
    # 예측용 입력 데이터 생성(100개 샘플)
    x0_range = np.linspace(x_train[:, 0].min(), x_train[:, 0].max(), 100)
    x1_range = np.linspace(x_train[:, 1].min(), x_train[:, 1].max(), 100)
    x_test_input = np.column_stack([x0_range, x1_range])  # (100, 2)

    # 예측
    y_predict = model.predict(x_test_input, verbose=0)

    ax0 = plt.subplot(5, 2, i*2 + 1)
    ax1 = plt.subplot(5, 2, i*2 + 2)
    plot_vision(ax0, ax1, x_train[:, 0], x0_range, x_val[:, 0], x_train[:, 1], x1_range, x_val[:, 1], 
                y_train[:, i], y_val[:, i], y_predict[:, i], y_predict[:, i], plant_names[i])
plt.tight_layout()
plt.show()

# %% 버전 1: 기준 이외 원인 고정 O, 역정규화 X
fig = plt.figure(figsize=(20, 12))
for i in range(5):
    # 광량만 변화, 시간은 중간값으로 고정
    x0_range = np.linspace(x_train[:, 0].min(), x_train[:, 0].max(), 100)
    fixed_time = (x_train[:, 1].min() + x_train[:, 1].max()) / 2
    x_test_input_0 = np.column_stack([x0_range, np.full(100, fixed_time)])
    
    # 시간만 변화, 광량은 중간값으로 고정
    x1_range = np.linspace(x_train[:, 1].min(), x_train[:, 1].max(), 100)
    fixed_light = (x_train[:, 0].min() + x_train[:, 0].max()) / 2
    x_test_input_1 = np.column_stack([np.full(100, fixed_light), x1_range])
    
    # 예측
    y_predict_0 = model.predict(x_test_input_0, verbose=0)
    y_predict_1 = model.predict(x_test_input_1, verbose=0)
    
    ax0 = plt.subplot(5, 2, i*2 + 1)
    ax1 = plt.subplot(5, 2, i*2 + 2)
    plot_vision(ax0, ax1, x_train[:, 0], x0_range, x_val[:, 0], x_train[:, 1], x1_range, x_val[:, 1], 
                y_train[:, i], y_val[:, i], y_predict_0[:, i], y_predict_1[:, i], plant_names[i])
    
plt.tight_layout()
plt.show()

# %% 버전 2: 기준 이외 원인 고정 X, 역정규화 O
fig = plt.figure(figsize=(20, 12))
for i in range(5):
    # 예측용 입력 데이터 생성(100개 샘플)
    x0_range = np.linspace(x_train[:, 0].min(), x_train[:, 0].max(), 100)
    x1_range = np.linspace(x_train[:, 1].min(), x_train[:, 1].max(), 100)
    x_test_input = np.column_stack([x0_range, x1_range])
    
    # 예측
    y_predict = model.predict(x_test_input, verbose=0)
    
    # 역정규화
    x0_range_original = x0_range * (x_max_0 - x_min_0) + x_min_0
    x1_range_original = x1_range * (x_max_1 - x_min_1) + x_min_1
    x_train_0_original = x_train[:, 0] * (x_max_0 - x_min_0) + x_min_0
    x_train_1_original = x_train[:, 1] * (x_max_1 - x_min_1) + x_min_1
    x_val_0_original = x_val[:, 0] * (x_max_0 - x_min_0) + x_min_0
    x_val_1_original = x_val[:, 1] * (x_max_1 - x_min_1) + x_min_1
    
    y_predict_original = y_predict[:, i] * (eval(f'y_max_{i}') - eval(f'y_min_{i}')) + eval(f'y_min_{i}')
    y_train_original = y_train[:, i] * (eval(f'y_max_{i}') - eval(f'y_min_{i}')) + eval(f'y_min_{i}')
    y_val_original = y_val[:, i] * (eval(f'y_max_{i}') - eval(f'y_min_{i}')) + eval(f'y_min_{i}')
    
    ax0 = plt.subplot(5, 2, i*2 + 1)
    ax1 = plt.subplot(5, 2, i*2 + 2)
    plot_vision(ax0, ax1, x_train_0_original, x0_range_original, x_val_0_original, x_train_1_original, x1_range_original, x_val_1_original, 
                y_train_original, y_val_original, y_predict_original, y_predict_original, plant_names[i])

plt.tight_layout()
plt.show()

# %% 버전 3: 기준 이외 원인 고정 O, 역정규화 O
fig = plt.figure(figsize=(20, 12))
for i in range(5):
    # 광량만 변화, 시간은 중간값으로 고정
    x0_range = np.linspace(x_train[:, 0].min(), x_train[:, 0].max(), 100)
    fixed_time = (x_train[:, 1].min() + x_train[:, 1].max()) / 2
    x_test_input_0 = np.column_stack([x0_range, np.full(100, fixed_time)])
    
    # 시간만 변화, 광량은 중간값으로 고정
    x1_range = np.linspace(x_train[:, 1].min(), x_train[:, 1].max(), 100)
    fixed_light = (x_train[:, 0].min() + x_train[:, 0].max()) / 2
    x_test_input_1 = np.column_stack([np.full(100, fixed_light), x1_range])
    
    # 예측
    y_predict_0 = model.predict(x_test_input_0, verbose=0)
    y_predict_1 = model.predict(x_test_input_1, verbose=0)
    
    # 역정규화
    x0_range_original = x0_range * (x_max_0 - x_min_0) + x_min_0
    x1_range_original = x1_range * (x_max_1 - x_min_1) + x_min_1
    fixed_time_original = fixed_time * (x_max_1 - x_min_1) + x_min_1
    fixed_light_original = fixed_light * (x_max_0 - x_min_0) + x_min_0
    
    x_train_0_original = x_train[:, 0] * (x_max_0 - x_min_0) + x_min_0
    x_train_1_original = x_train[:, 1] * (x_max_1 - x_min_1) + x_min_1
    x_val_0_original = x_val[:, 0] * (x_max_0 - x_min_0) + x_min_0
    x_val_1_original = x_val[:, 1] * (x_max_1 - x_min_1) + x_min_1
    
    y_predict_0_original = y_predict_0[:, i] * (eval(f'y_max_{i}') - eval(f'y_min_{i}')) + eval(f'y_min_{i}')
    y_predict_1_original = y_predict_1[:, i] * (eval(f'y_max_{i}') - eval(f'y_min_{i}')) + eval(f'y_min_{i}')
    y_train_original = y_train[:, i] * (eval(f'y_max_{i}') - eval(f'y_min_{i}')) + eval(f'y_min_{i}')
    y_val_original = y_val[:, i] * (eval(f'y_max_{i}') - eval(f'y_min_{i}')) + eval(f'y_min_{i}')
    
    ax0 = plt.subplot(5, 2, i*2 + 1)
    ax1 = plt.subplot(5, 2, i*2 + 2)
    plot_vision(ax0, ax1, x_train_0_original, x0_range_original, x_val_0_original, x_train_1_original, x1_range_original, x_val_1_original, 
                y_train_original, y_val_original, y_predict_0_original, y_predict_1_original, plant_names[i])

plt.tight_layout()

plt.tight_layout()
plt.show()

# %% 손실 그래프 시각화
plt.figure(figsize=(12, 5))

# 학습 손실과 검증 손실
plt.subplot(1, 2, 1)
plt.plot(hist.history['loss'], label='Training Loss', linewidth=2)
plt.plot(hist.history['val_loss'], label='Validation Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Model Training History')
plt.legend()
plt.grid(True, alpha=0.3)

# 로그 스케일로 표시
plt.subplot(1, 2, 2)
plt.plot(hist.history['loss'], label='Training Loss', linewidth=2)
plt.plot(hist.history['val_loss'], label='Validation Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Model Training History (Log Scale)')
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% 최종 손실 출력
print(f"최종 학습 손실: {hist.history['loss'][-1]:.6f}")
print(f"최종 검증 손실: {hist.history['val_loss'][-1]:.6f}")
print(f"최소 검증 손실: {min(hist.history['val_loss']):.6f} (Epoch {np.argmin(hist.history['val_loss']) + 1})")

# %%
