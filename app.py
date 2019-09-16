from flask import Flask
from flask_restful import Api
from resources.meshes_deep3d_face import MeshesDeep3DFaceGetMesh

app = Flask(__name__)
api = Api(app)

api.add_resource(MeshesDeep3DFaceGetMesh,
                 '/meshes-deep3d-face/reconstruct-mesh')

if __name__ == '__main__':
    app.run()
