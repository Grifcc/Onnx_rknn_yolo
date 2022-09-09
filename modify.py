import onnx
import onnx_graphsurgeon as gs
import numpy as np
import onnxsim

src_onnx_path = "/workspace/yolov7_rknn/model/onnx_ori/yolov7-tiny-visdrone.onnx"
dst_onnx_path = "/workspace/yolov7_rknn/model/onnx_modify/yolov7-tiny-visdrone_rm_reshape.onnx"

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

if(len(graph.outputs) !=4):
    raise ValueError

    
# nodes_dict['Conv_344'].outputs[0].dtype = np.float32
# nodes_dict['Conv_374'].outputs[0].dtype = np.float32
# nodes_dict['Conv_404'].outputs[0].dtype = np.float32

nodes_dict['Conv_134'].outputs[0].shape = (1,45,80,80)
nodes_dict['Conv_164'].outputs[0].shape = (1,45,40,40)
nodes_dict['Conv_194'].outputs[0].shape = (1,45,20,20)

graph.outputs[0] = nodes_dict['Conv_134'].outputs[0]
graph.outputs[1] = nodes_dict['Conv_164'].outputs[0]
graph.outputs[2] = nodes_dict['Conv_194'].outputs[0]
graph.outputs.pop(3)

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