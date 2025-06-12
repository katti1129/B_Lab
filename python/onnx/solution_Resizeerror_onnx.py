import onnx

def fix_split_ops(model):
    for i in range(len(model.graph.node)):
        node = model.graph.node[i]
        if node.op_type != 'Split':
            continue
        
        # 出力ノード数に基づいてsplit引数を設定
        split = (1, ) * len(node.output)
        new_node = onnx.helper.make_node('Split', node.input, node.output, split=split)
        model.graph.node.insert(i, new_node)
        model.graph.node.remove(node)

# エクスポートされたONNXモデルをロード
model_path = "/Users/katti/Desktop/Lab/python/yolov8n-2.onnx"
onnx_model = onnx.load(model_path)

# Splitオペレーションを修正
fix_split_ops(onnx_model)

# 修正後のモデルを保存
onnx.save(onnx_model, 'yolov8n-split-fixed.onnx')
