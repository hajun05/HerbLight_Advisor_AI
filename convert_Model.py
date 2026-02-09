# %% 
import numpy as np
import tf2onnx
import onnx
import tensorflow as tf

# %% 모델 불러오기, 사용하기
loaded_model = tf.keras.models.load_model("HerbLight_Advisor_model.keras")
loaded_model.summary()

# %% 모델을 ONNX 포맷으로 변환, 저장
spec = (tf.TensorSpec((None, 2), tf.float32, name="input"),)
loaded_model.output_names = ["output"] # 출력 이름 지정해야 변환 가능
onnx_model, _ = tf2onnx.convert.from_keras(loaded_model, input_signature=spec, opset=13)
onnx.save(onnx_model, "HerbLight_Advisor_model.onnx")


# %%
