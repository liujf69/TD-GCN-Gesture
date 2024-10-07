'''
@File    :   export_onnx.py
@Time    :   2024/10/07 19:30:00
@Author  :   Jinfu Liu
@Version :   1.0 
@Desc    :   Code demo of export onnx
'''

import sys
import torch
import traceback

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))
    
    
if __name__ == "__main__":
    # init model
    graph = "graph.shrec17.Graph" # shrec_17
    graph_args = {'labeling_mode': 'spatial'}
    Model = import_class('model.tdgcn.Model')
    num_class = 28 # shrec_17
    num_point = 22 # shrec_17
    num_person = 1 # shrec_17
    model = Model(num_class = num_class, num_point = num_point, num_person = num_person, graph = graph, graph_args = graph_args)
    
    # load weight
    model_dict = model.state_dict() 
    weights_files = './checkpoints/Shrec17/28label_joint.pt'
    weights = torch.load(weights_files) 
    match_dict = {k: v for k, v in weights.items() if k in model_dict}
    model_dict.update(match_dict)
    model.load_state_dict(model_dict)
    model.eval()
    
    # init input
    B = 32
    C = 3
    T = 180 # shrec_17
    V = 22 # # shrec_17
    M = 1 # # shrec_17
    input_data = torch.rand(B, C, T, V, M)
    
    # export onnx
    input_name = 'input'
    output_name = 'output'
    torch.onnx.export(model, 
                        input_data, 
                        "./Static_demo.onnx", 
                        verbose = True, 
                        input_names = [input_name], 
                        output_names = [output_name]
                    )
    
    torch.onnx.export(model, 
                        input_data, 
                        "./Dynamics_demo.onnx",
                        opset_version = 12,
                        input_names = [input_name],
                        output_names = [output_name],
                        dynamic_axes = {
                            input_name: {0: 'batch_size', 2: 'input_frames', 3: 'num_joints', 4: 'num_bodies'},
                            output_name: {0: 'batch_size', 1: 'num_classes'}
                        }
                    )
    print("All Done!")