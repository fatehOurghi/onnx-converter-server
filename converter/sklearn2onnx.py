import onnxmltools

def sklearn2onnx(args):
    from sklearn.externals import joblib
    from skl2onnx import convert_sklearn
    # Check for required arguments
    if not args.get("initial_types"):
        raise ValueError("Please provide --initial_types to convert scikit learn models.")
    # Load your sklearn model
    skl_model = joblib.load(args.get("model"))
    
    # Convert the sklearn model into ONNX
    onnx_model = onnxmltools.convert_sklearn(skl_model, 
        initial_types = args.get("initial_types"),
        target_opset=int(args.get("target_opset")))
    # Save as protobuf
    onnxmltools.utils.save_model(onnx_model, args.get("output_onnx_path"))

