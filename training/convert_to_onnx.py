import joblib
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from onnxruntime import InferenceSession
from onnxruntime.quantization import quantize_dynamic, QuantType

# -----------------------------
# Step 1: Load trained MLP
# -----------------------------
mlp_model_path = "Model/mlp_model_best.pkl"
mlp_model = joblib.load(mlp_model_path)

# -----------------------------
# Step 2: Convert to ONNX
# -----------------------------
onnx_model_path = "Model/Weak_up_model.onnx"
initial_type = [('input', FloatTensorType([None, 768]))]  # adjust 768 to your feature size
onnx_model = convert_sklearn(mlp_model, initial_types=initial_type, target_opset=17)

with open(onnx_model_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

# -----------------------------
# Step 3: Apply dynamic INT8 quantization
# -----------------------------
onnx_int8_path = "Model/mlp_model_best_int8.onnx"
quantize_dynamic(
    model_input=onnx_model_path,
    model_output=onnx_int8_path,
    weight_type=QuantType.QInt8
)

# -----------------------------
# Step 4: Safe ONNX inference
# -----------------------------
def run_onnx_inference(onnx_path, X):
    """
    X: numpy array, shape = (n_samples, n_features)
    Returns predicted class indices
    """
    session = InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    pred = session.run(None, {input_name: X.astype(np.float32)})[0]

    # Handle 1D or 2D output safely
    if pred.ndim == 1:
        pred_classes = pred.astype(int)
    else:
        pred_classes = np.argmax(pred, axis=1)
    return pred_classes

# -----------------------------
# Step 5: Test and compare
# -----------------------------
X_test = np.random.rand(5, 768).astype(np.float32)  # example features

sklearn_pred = mlp_model.predict(X_test)
onnx_pred_classes = run_onnx_inference(onnx_int8_path, X_test)

print("Sklearn MLP prediction: ", sklearn_pred)
print("ONNX INT8 prediction:   ", onnx_pred_classes)
