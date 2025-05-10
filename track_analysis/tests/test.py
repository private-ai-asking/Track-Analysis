# # export_quantize_miniLM.py
# # Requires: pip install transformers onnx onnxruntime onnxruntime-tools
#
# from pathlib import Path
# import os
#
# from transformers import AutoModel, AutoTokenizer
# from transformers.models.bert import BertOnnxConfig
# from transformers.onnx.convert import export
# from onnxruntime.quantization import quantize_dynamic, QuantType
#
# # --- Configurable paths & names ---
# MODEL_NAME     = "sentence-transformers/all-MiniLM-L6-v2"
# OPSET          = 18
# ONNX_DIR       = Path("onnx_models")
# ONNX_FP32      = ONNX_DIR / "miniLM.onnx"
# ONNX_INT8      = ONNX_DIR / "miniLM_int8.onnx"
# DEVICE         = "cuda"    # or "cuda" if you want to export via GPU
#
# ONNX_DIR.mkdir(exist_ok=True, parents=True)
#
# # 1) Load original model + tokenizer
# print(f"Loading model & tokenizer: {MODEL_NAME}")
# model     = AutoModel.from_pretrained(MODEL_NAME)
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#
# # 2) Prepare the correct ONNX config subclass for a BERT-style model
# onnx_config = BertOnnxConfig(model.config)
#
# # 3) Export to ONNX (FP32)
# print(f"Exporting to ONNX at {ONNX_FP32}")
# # export signature:
# #   export(preprocessor, model, config, opset, output, tokenizer=None, device='cpu')
# export(
#     tokenizer,     # preprocessor (AutoTokenizer)
#     model,         # model instance
#     onnx_config,   # BertOnnxConfig
#     OPSET,         # ONNX opset version
#     ONNX_FP32,     # output file path
#     device=DEVICE  # device for export
# )
# print("✅ ONNX FP32 export done.")
#
# # 4) Quantize to INT8
# print(f"Quantizing to INT8 at {ONNX_INT8}")
# quantize_dynamic(
#     model_input=str(ONNX_FP32),
#     model_output=str(ONNX_INT8),
#     per_channel=False,
#     weight_type=QuantType.QInt8
# )
# print("✅ ONNX INT8 quantization done.")
#
# # 5) Report sizes
# for path in (ONNX_FP32, ONNX_INT8):
#     size_mb = os.path.getsize(path) / (1024*1024)
#     print(f" • {path.name}: {size_mb:.2f} MB")
