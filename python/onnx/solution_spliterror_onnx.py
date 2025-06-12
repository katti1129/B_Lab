"""
#mymodelをonnxに変換.splitエラー回避．
# splitエラー回避策
# https://learning.unity3d.jp/7557/
import onnx
# Splitエラー対策
def fix_split_ops(model):
    for i in range(len(model.graph.node)):
        node = model.graph.node[i]
        if node.op_type != 'Split':
            continue

        split = (1, ) * len(node.output)
        new_node = onnx.helper.make_node('Split', node.input, node.output, split = split)
        model.graph.node.insert(i, new_node)
        model.graph.node.remove(node)
# onnx_model is an in-memory ModelProto
model_path = "./train4/weights/best.onnx"
onnx_model = onnx.load(model_path)
fix_split_ops(onnx_model)
save_path = "./train4/weights/best_2.onnx"
onnx.save(onnx_model, save_path)
"""

# splitエラー回避策
#officialのyolov8nモデルエラー回避
# https://learning.unity3d.jp/7557/
import onnx
# Splitエラー対策
def fix_split_ops(model):
    for i in range(len(model.graph.node)):
        node = model.graph.node[i]
        if node.op_type != 'Split':
            continue

        split = (1, ) * len(node.output)
        new_node = onnx.helper.make_node('Split', node.input, node.output, split = split)
        model.graph.node.insert(i, new_node)
        model.graph.node.remove(node)
# onnx_model is an in-memory ModelProto
model_path = "/Users/katti/Desktop/Lab/python/yolov8n.onnx"
onnx_model = onnx.load(model_path)
fix_split_ops(onnx_model)
onnx.save(onnx_model, 'yolov8n-2.onnx')
