import numpy as np
def quantize_onnx_model(onnx_model_path, quantized_model_path):
    from onnxruntime.quantization.quantize import quantize_dynamic, QuantType
    import onnx
    onnx_opt_model = onnx.load(onnx_model_path)
    quantize_dynamic(onnx_model_path,
                     quantized_model_path,
                     weight_type=QuantType.QInt8)

    print(f"quantized model saved to:{quantized_model_path}")



quantize_onnx_model("model.onnx", "model_quant.onnx")
