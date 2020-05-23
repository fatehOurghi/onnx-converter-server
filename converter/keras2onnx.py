import onnxmltools


def keras2onnx(args):
    import keras
    # Load your Keras model
    keras_model = keras.models.load_model(args.model)

    # Convert the Keras model into ONNX
    onnx_model = onnxmltools.convert_keras(keras_model, 
        initial_types = args.initial_types,
        target_opset=int(args.target_opset))

    # Save as protobuf
    onnxmltools.utils.save_model(onnx_model, args.output_onnx_path)

