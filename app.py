from flask import Flask
from flask_restful import Api
from resources.meshes_deep3d_face import MeshesDeep3DFaceGetMesh
import argparse

app = Flask(__name__)
api = Api(app)

api.add_resource(MeshesDeep3DFaceGetMesh,
                 '/meshes-deep3d-face/reconstruct-mesh')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Deep3DFace Server')

    parser.add_argument('--ip', default='0.0.0.0', type=str,
                        help='ip address of the service')

    parser.add_argument('--port', default='5000', type=str,
                        help='port of the service')

    args = parser.parse_args()

    app.run(host=args.ip, port=args.port)
