from ultralytics import YOLO
import onnx
#Barracuda用に引数アリにしか対応していないのでSplit
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
model = YOLO('yolov8n.pt')
model.export(format='onnx', opset=12, simplify=True)
onnx_model = onnx.load('yolov8n.onnx')
fix_onnx_resize(onnx_model)
onnx.save(onnx_model, 'yolov8n-barracuda.onnx')
