"""
import onnx
def fix_onnx_resize(model):
    for i in range(len(model.graph.node)):
        node = model.graph.node[i]
        if node.op_type != 'Resize':
            continue
        
        new_node = onnx.helper.make_node(
                'Resize',
                inputs=node.input,
                outputs=node.output,
                name=node.name,
                coordinate_transformation_mode='half_pixel',  # Instead of pytorch_half_pixel, unsupported by Tensorflow
                mode='linear',
            )
        model.graph.node.insert(i, new_node)
        model.graph.node.remove(node)
        
model_path = "./train4/weights/best_2.onnx"
onnx_model = onnx.load(model_path)
fix_onnx_resize(onnx_model)
save_path = "./train4/weights/best_3.onnx"
onnx.save(onnx_model, save_path)
"""
import onnx
import onnx.helper

def fix_onnx_resize(model):
    for i in range(len(model.graph.node)):
        node = model.graph.node[i]
        if node.op_type != 'Resize':
            continue
        
        new_node = onnx.helper.make_node(
            'Resize',
            inputs=node.input,
            outputs=node.output,
            name=node.name,
            coordinate_transformation_mode='half_pixel',
            mode='linear',
        )
        model.graph.node.insert(i, new_node)
        model.graph.node.remove(node)

def set_input_shape(model):
    for input in model.graph.input:
        shape = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
        if shape == [0]:  # デフォルトの形状が不明な場合
            input.type.tensor_type.shape.dim[0].dim_value = 1  # バッチサイズ
            input.type.tensor_type.shape.dim[1].dim_value = 3  # チャンネル数
            input.type.tensor_type.shape.dim[2].dim_value = 224  # 高さ
            input.type.tensor_type.shape.dim[3].dim_value = 224  # 幅

model_path = "./train4/weights/best_2.onnx"
save_path = "./train4/weights/best_4.onnx"

onnx_model = onnx.load(model_path)
fix_onnx_resize(onnx_model)
set_input_shape(onnx_model)
onnx.save(onnx_model, save_path)

