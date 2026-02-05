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
    [100, 8],   # 낮은 광량, 짧은 시간
    [200, 10],  # 중간 광량, 중간 시간
    [300, 12],  # 중간 광량, 긴 시간
    [400, 14],  # 높은 광량, 긴 시간
    [150, 6],   # 매우 낮은 광량, 짧은 시간
    [250, 9],   # 중간 광량, 중간 시간
    [350, 11],  # 높은 광량, 중간 시간
    [500, 16],  # 매우 높은 광량, 매우 긴 시간
    [180, 7],   # 낮은 광량, 짧은 시간
    [320, 13]   # 높은 광량, 긴 시간
], dtype=float)

y_val = np.array([
    [55, 48, 62, 50, 58],    # 낮은 광량, 짧은 시간
    [78, 72, 85, 75, 80],    # 중간 광량, 중간 시간
    [95, 88, 102, 92, 98],   # 중간-높은 광량, 긴 시간
    [115, 108, 125, 112, 118],  # 높은 광량, 긴 시간
    [42, 38, 50, 40, 45],    # 매우 낮은 광량, 짧은 시간
    [68, 62, 75, 65, 70],    # 중간 광량, 중간 시간
    [88, 82, 98, 85, 92],    # 높은 광량, 중간 시간
    [145, 138, 155, 142, 148],  # 매우 높은 광량, 매우 긴 시간
    [50, 45, 58, 48, 52],    # 낮은 광량, 짧은 시간
    [105, 98, 115, 102, 108]    # 높은 광량, 긴 시간
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

# %%
