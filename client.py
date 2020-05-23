import requests
import json
import os

url = "http://localhost:8001"


def converter_client(url):
    # prepare data
    data_to_send = {
        "model": "models/model.h5",
        "model_type": "keras",
        "output_onnx_path": "models/model.onnx",
        "test_data_path": "",
        "model_inputs_names": "",
        "model_outputs_names": "",
        "model_input_shapes": "",
        "initial_types": "",
        "target_opset": "",
        "caffe_model_prototxt": ""
    }
    headers = {
        'content-type': 'application/json'
    }

    # send to server and receive response
    response = requests.post(url, json=data_to_send, headers=headers)

    # postprocess
    data = json.loads(response.text)

    
    print(data)


converter_client(url)
