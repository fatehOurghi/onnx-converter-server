import coremltools
import onnxmltools

def coreml2onnx(args):
    # Load your CoreML model
    coreml_model = coremltools.utils.load_spec(args.get("model"))

    # Convert the CoreML model into ONNX
    onnx_model = onnxmltools.convert_coreml(coreml_model, 
        initial_types = args.get("initial_types"),
        target_opset=int(args.get("target_opset")))

    # Save as protobuf
    onnxmltools.utils.save_model(onnx_model, args.get("output_onnx_path"))
