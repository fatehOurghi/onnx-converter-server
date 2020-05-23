import onnxmltools


def pytorch2onnx(args):
    # PyTorch exports to ONNX without the need for an external converter
    import torch
    from torch.autograd import Variable
    import torch.onnx
    import torchvision
    # Create input with the correct dimensions of the input of your model
    if args.get("model_input_shapes") == None:
        raise ValueError("Please provide --model_input_shapes to convert Pytorch models.")
    dummy_model_input = []
    if len(args.get("model_input_shapes")) == 1:
        dummy_model_input = Variable(torch.randn(*args.get("model_input_shapes")))
    else:
        for shape in args.get("model_input_shapes"):
            dummy_model_input.append(Variable(torch.randn(*shape)))

    # load the PyTorch model
    model = torch.load(args.get("model"), map_location="cpu")

    # export the PyTorch model as an ONNX protobuf
    torch.onnx.export(model, dummy_model_input, args.get("output_onnx_path"))

