import subprocess
from .check_model import get_extension, check_model
import onnxmltools



def tf2onnx(args): 
    if get_extension(args.get("model")) == "pb":
        if not args.get("model_inputs_names") and not args.get("model_outputs_names"):
            raise ValueError("Please provide --model_inputs_names and --model_outputs_names to convert Tensorflow graphdef models.")
        subprocess.check_call(["python", "-m", "tf2onnx.convert", 
            "--input", args.get("model"), 
            "--output", args.get("output_onnx_path"), 
            "--inputs", args.get("model_inputs_names"),
            "--outputs", args.get("model_outputs_names"), 
            "--opset", args.get("target_opset"), 
            "--fold_const",
            "--target", "rs6"])
    elif get_extension(args.get("model")) == "meta":
        if not args.get("model_inputs_names") and not args.get("model_outputs_names"):
            raise ValueError("Please provide --model_inputs_names and --model_outputs_names to convert Tensorflow checkpoint models.")
        subprocess.check_call(["python", "-m", "tf2onnx.convert", 
            "--checkpoint", args.get("model"), 
            "--output", args.get("output_onnx_path"), 
            "--inputs", args.get("model_inputs_names"),
            "--outputs", args.get("model_outputs_names"), 
            "--opset", args.get("target_opset"), 
            "--fold_const",
            "--target", "rs6"])
    else:
        subprocess.check_call(["python", "-m", "tf2onnx.convert", 
            "--saved-model", args.get("model"), 
            "--output", args.get("output_onnx_path"), 
            "--opset", args.get("target_opset"),
            "--fold_const",
            "--target", "rs6"])

