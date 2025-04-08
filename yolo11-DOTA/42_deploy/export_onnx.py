from ultralytics import YOLO

# Load a model
model = YOLO("../42_demo/yolo11n.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom trained model

# Export the model
# model.export(format="onnx", dynamic=True, int8=True)
# model.export(format="engine", int8=True)
model.export(format="mnn")
# model.export(format="onnx", half=True)
# model.export(format="ncnn")
# tensorflow模型导出有问题
# ['tf_keras', 'sng4onnx>=1.0.1', 'onnx_graphsurgeon>=0.3.26', 'onnx2tf>1.17.5,<=1.22.3', 'tflite_support'] not found, attempting AutoUpdate...
'''
format: 导出模型的目标格式（例如："......"）、 onnx, torchscript, tensorflow).
imgsz: 模型输入所需的图像大小（例如："......"）、 640 或 (height, width)).
half: 启用 FP16 量化，减少模型大小，并可能加快推理速度。
optimize: 针对移动或受限环境进行特定优化。
int8: 启用 INT8 量化，非常有利于边缘部署。
'''

# 安卓文件
# 安卓文件的版本选择2022.2.1