import onnx
import onnx_graphsurgeon as gs
import numpy as np
import onnxsim

src_onnx_path = "/workspace/Onnx_rknn_yolo/model/onnx_ori/yolov5s-visdrone-feax2-qat5.onnx"
dst_onnx_path = "/workspace/Onnx_rknn_yolo/model/onnx_modify/yolov5s-visdrone-feax2-qat5_rm_reshape.onnx"

print("Load onnx model from {}".format(src_onnx_path))
graph = gs.import_onnx(onnx.load(src_onnx_path))
print("Nodes:{}".format(len(graph.nodes)))
graph.fold_constants().cleanup()
nodes = graph.nodes

nodes_dict = {}
for node in nodes:
    name = node.name
    nodes_dict.update({name: node})



print(len(graph.outputs))

# if(len(graph.outputs) !=4):
#     raise ValueError

    
nodes_dict['Conv_198'].outputs[0].dtype=graph.outputs[0].dtype
nodes_dict['Conv_214'].outputs[0].dtype=graph.outputs[0].dtype
nodes_dict['Conv_230'].outputs[0].dtype=graph.outputs[0].dtype

nodes_dict['Conv_198'].outputs[0].shape = (1,45,160,160)
nodes_dict['Conv_214'].outputs[0].shape = (1,45,80,80)
nodes_dict['Conv_230'].outputs[0].shape = (1,45,40,40)

graph.outputs[0] = nodes_dict['Conv_198'].outputs[0]
graph.outputs[1] = nodes_dict['Conv_214'].outputs[0]
graph.outputs[2] = nodes_dict['Conv_230'].outputs[0]
# graph.outputs.pop(3)

print(len(graph.outputs))
# graph.outputs[0]=nodes_dict['Conv_366'].outputs[0]
# graph.outputs.append(nodes_dict['Conv_336'].outputs[0])
# graph.outputs.append(nodes_dict['Conv_306'].outputs[0])

graph.cleanup().toposort()
print("Nodes:{}".format(len(graph.nodes)))

print('\nStarting to simplify ONNX...')
onnx_model, check = onnxsim.simplify(gs.export_onnx(graph))
onnx.save(onnx_model, dst_onnx_path)
print("Save modified onnx model to {}".format(dst_onnx_path))