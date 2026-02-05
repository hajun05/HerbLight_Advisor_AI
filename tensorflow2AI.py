#%% 패키지 설치
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense

#%% 학습용 엑셀 데이터 불러오기
# 학습용 데이터 (training set)
x_train = pd.read_excel("dataset\\허브 광량 환경 데이터셋.xlsx", sheet_name="원인").to_numpy().astype(float)
y_train = pd.read_excel("dataset\\허브 광량 환경 데이터셋.xlsx", sheet_name="결과").to_numpy().astype(float)

#%% min-max 정규화


# %%
