import os
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

MODELS_DIR = "models"

# INT8 quantization candidates (NOT face detection/recognition — biometric accuracy critical)
INT8_MODELS = [
    "yolo26m.onnx",
    "dinov2_vits14.onnx",
]

def quantize_model_int8(model_name):
    input_path = os.path.join(MODELS_DIR, model_name)
    output_path = os.path.join(MODELS_DIR, model_name.replace(".onnx", "_int8.onnx"))

    if not os.path.exists(input_path):
        print(f"Skipping INT8 {model_name}: File not found.")
        return

    print(f"Quantizing {model_name} -> {output_path} (INT8)...")

    try:
        quantize_dynamic(
            model_input=input_path,
            model_output=output_path,
            weight_type=QuantType.QUInt8
        )
        print(f"Successfully quantized {model_name} to INT8.")
    except Exception as e:
        print(f"Failed to quantize {model_name}: {e}")

if __name__ == "__main__":
    if not os.path.exists(MODELS_DIR):
        print(f"Error: '{MODELS_DIR}' directory not found.")
        exit(1)

    print("=== INT8 Quantization (Optional — for person detection and ReID models only) ===")
    print("Note: Face detection (det.onnx) and recognition (rec.onnx) are NOT quantized")
    print("      to preserve biometric accuracy.\n")
    for model in INT8_MODELS:
        quantize_model_int8(model)

    print("\nDone! To use quantized models, rename the '_int8.onnx' versions to replace the originals.")
