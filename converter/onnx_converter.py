# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import argparse
import subprocess
from pathlib import Path
from shutil import copyfile
from .check_model import get_extension, check_model
from .create_input import generate_inputs
import coremltools
import onnxmltools
from onnxmltools.convert.common.data_types import *
import json
import os
import pprint

from .tf2onnx import tf2onnx
from .caffe2onnx import caffe2onnx
from .cntk2onnx import cntk2onnx
from .coreml2onnx import coreml2onnx
from .keras2onnx import keras2onnx
from .pytorch2onnx import pytorch2onnx
from .sklearn2onnx import sklearn2onnx


class ConverterParamsFromJson():
    def __init__(self, input_json):
        # Check the required inputs
        if input_json.get("model") == None:
            raise ValueError("Please specified \"model\" in the input json. ")
        if input_json.get("model_type") == None:
            raise ValueError("Please specified \"model_type\" in the input json. ")
        if input_json.get("output_onnx_path") == None:
            raise ValueError("Please specified \"output_onnx_path\" in the input json. ")

        self.model = input_json["model"]
        self.model_type = input_json["model_type"]
        self.test_data_path = input_json["test_data_path"] if input_json.get("test_data_path") else None
        self.output_onnx_path = input_json["output_onnx_path"]
        self.model_inputs_names = input_json["model_inputs_names"] if input_json.get("model_inputs_names") else None
        self.model_outputs_names = input_json["model_outputs_names"] if input_json.get("model_outputs_names") else None
        self.model_input_shapes = shape_type(input_json["model_input_shapes"]) if input_json.get("model_input_shapes") else None
        self.initial_types = eval(input_json["initial_types"]) if input_json.get("initial_types") else None
        self.target_opset = input_json["target_opset"] if input_json.get("target_opset") else "10"
        self.caffe_model_prototxt = input_json["caffe_model_prototxt"] if input_json.get("caffe_model_prototxt") else None


def shape_type(s):
    import ast
    if s == None or len(s) == 0:
        return
    try:
        shapes_list = list(ast.literal_eval(s))
        if isinstance(shapes_list[0], tuple) == False:
            # Nest the shapes list to make it a list of tuples
            return [tuple(shapes_list)]
        return shapes_list
    except:
        raise argparse.ArgumentTypeError("Model input shapes must be a list of tuple. Each dimension separated by ','. ")


suffix_format_map = {
    "h5": "keras",
    "keras": "keras",
    "mlmodel": "coreml",
}


converters = {
    "caffe": caffe2onnx,
    "cntk": cntk2onnx,
    "coreml": coreml2onnx,
    "keras": keras2onnx,
    "scikit-learn": sklearn2onnx,
    "pytorch": pytorch2onnx,
    "tensorflow": tf2onnx
}


output_template = {
    "output_onnx_path": "", # The output path where the converted .onnx file is stored. 
    "conversion_status": "", # SUCCEED, FAILED
    "correctness_verified": "", # SUCCEED, NOT SUPPORTED, SKIPPED
    "input_folder": "", 
    "error_message": ""
}


def convert_models(args):
    # Quick format check
    model_extension = get_extension(args.model)
    if (args.model_type == "onnx" or model_extension == "onnx"):
        print("Input model is already ONNX model. Skipping conversion.")
        if args.model != args.output_onnx_path:
            copyfile(args.model, args.output_onnx_path)
        return
    
    if converters.get(args.model_type) == None:
        raise ValueError('Model type {} is not currently supported. \n\
            Please select one of the following model types -\n\
                cntk, coreml, keras, pytorch, scikit-learn, tensorflow'.format(args.model_type))
    
    suffix = suffix_format_map.get(model_extension)

    if suffix != None and suffix != args.model_type:
        raise ValueError('model with extension {} do not come from {}'.format(model_extension, args.model_type))

    # Find the corresponding converter for current model
    converter = converters.get(args.model_type)
    # Run converter
    converter(args)



def convert(input_json):        
    if input_json != None and len(input_json) > 0:
        args = ConverterParamsFromJson(input_json)
    else:
        if not args.model or len(args.model) == 0:
            raise ValueError("Please specify the required argument \"model\" either in a json file or by --model")
        if not args.model_type or len(args.model_type) == 0:
            raise ValueError("Please specify the required argument \"model_type\" either in a json file or by --model_type")
        if not args.output_onnx_path or len(args.output_onnx_path) == 0:
            raise ValueError("Please specify the required argument \"output_onnx_path\" either in a json file or by --ouptut_onnx_path")
        if args.initial_types and len(args.initial_types) > 0:
            args.initial_types = eval(args.initial_types)
    # Create a test folder path
    output_dir = os.path.dirname(os.path.abspath(args.output_onnx_path))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_json_path = os.path.join(output_dir, "output.json")
    print("\n-------------\nModel Conversion\n")
    try:
        convert_models(args)
    except Exception as e:
        print("Conversion error occurred. Abort. ")
        output_template["conversion_status"] = "FAILED"
        output_template["correctness_verified"] = "FAILED"
        output_template["error_message"] = str(e)
        print("\n-------------\nMODEL CONVERSION SUMMARY (.json file generated at %s )\n" % output_json_path)
        pprint.pprint(output_template)
        with open(output_json_path, "w") as f:
            json.dump(output_template, f, indent=4)
        raise e

    output_template["conversion_status"] = "SUCCESS"
    output_template["output_onnx_path"] = args.output_onnx_path

    print("\n-------------\nMODEL INPUT GENERATION(if needed)\n")
    # Generate random inputs for the model if input files are not provided
    try:
        inputs_path = generate_inputs(args.model, args.test_data_path, args.output_onnx_path)
        output_template["input_folder"] = inputs_path
    except Exception as e:
        output_template["error_message"]= str(e)
        output_template["correctness_verified"] = "SKIPPED"
        print("\n-------------\nMODEL CONVERSION SUMMARY (.json file generated at %s )\n" % output_json_path)
        pprint.pprint(output_template)
        with open(output_json_path, "w") as f:
            json.dump(output_template, f, indent=4)
        raise e

    print("\n-------------\nMODEL CORRECTNESS VERIFICATION\n")
    # Test correctness
    verify_status = check_model(args.model, args.output_onnx_path, inputs_path, args.model_type, args.model_inputs_names, args.model_outputs_names)
    output_template["correctness_verified"] = verify_status

    print("\n-------------\nMODEL CONVERSION SUMMARY (.json file generated at %s )\n" % output_json_path)
    pprint.pprint(output_template)
    with open(output_json_path, "w") as f:
        json.dump(output_template, f, indent=4)

