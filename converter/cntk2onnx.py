import coremltools
import onnxmltools

def cntk2onnx(args):
    import cntk
    # Load your CNTK model
    cntk_model = cntk.Function.load(args.get("model"), device=cntk.device.cpu())

    # Convert the CNTK model into ONNX
    cntk_model.save(args.get("output_onnx_path"), format=cntk.ModelFormat.ONNX)
