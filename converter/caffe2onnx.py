import coremltools
import onnxmltools


def caffe2onnx(args):
    caffe_model = args.get("model")
    # Convert Caffe model to CoreML 
    if args.get("caffe_model_prototxt") != None and len(args.get("caffe_model_prototxt"))> 0:
        caffe_model = (args.get("model"), args.get("caffe_model_prototxt"))
    coreml_model = coremltools.converters.caffe.convert(caffe_model)

    # Name and path for intermediate coreml model
    output_coreml_model = 'model.mlmodel'

    # Save CoreML model
    coreml_model.save(output_coreml_model)

    # Load a Core ML model
    coreml_model = coremltools.utils.load_spec(output_coreml_model)

    # Convert the Core ML model into ONNX
    onnx_model = onnxmltools.convert_coreml(coreml_model, target_opset=int(args.get("target_opset")))

    # Save as protobuf
    onnxmltools.utils.save_model(onnx_model, args.get("output_onnx_path"))
