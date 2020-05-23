from http.server import HTTPServer, BaseHTTPRequestHandler
import onnxruntime as rt
import numpy as np
import os
import json
import argparse

from converter.onnx_converter import convert

# this is the root folder of your models directory
models_dir = os.path.join(os.getcwd(),'models/')


class ONNXConverter(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        f = open(os.path.join(os.getcwd(), "index.html"), "rb").read()
        self.wfile.write(f)

    def do_POST(self):
        # parse request
        length = int(self.headers.get("Content-Length"))
        data = json.loads(self.rfile.read(length))

        # convert
        convert(data)

        # build response
        #data_to_send = {
        #    "output": output
        #}

        # response ok
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()

        # send data
        #self.wfile.write(json.dumps(data_to_send).encode(encoding='utf_8'))


def run(port):
    server = HTTPServer(("0.0.0.0", port), ONNXConverter)
    print(f'Server running on http://localhost:{port}')
    server.serve_forever()


parser = argparse.ArgumentParser(
    description="provide the port number as --port #port"
)
parser.add_argument('--port', type=int, default=8001, help="port number (integer)")
args = parser.parse_args()
port = args.port if args.port > 0 else 8001

run(port)
