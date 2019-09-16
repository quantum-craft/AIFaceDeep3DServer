# AI Face Morphing Deep 3D Face Reconstruction Server

Deep 3D Face Reconstruction Server for AI face morphing project

## Requirements

- Python 3.6.8
- Flask
- Dlib
- CMake for Dlib installation
- (Highly Recommended) CUDA capable GPU
- (Highly Recommended) CUDA toolkits 10.0

## Installation

- It is recommended to use Conda(Anaconda) to control Python environment.

### Create and Activate Conda Environment

```bash
conda create --name deep3d_face python=3.6.8
conda activate deep3d_face
pip install --upgrade pip
```

- It is also recommended to use Visual Studio Code to debug. Please see .vscode/launch.json
- Postman is a very good app to send test http request.

### Install Flask

```bash
pip install flask-restful  
```

### Install TensorFlow

- Deep3D Face Reconstruction requires TensorFlow >= 1.14
- For more information, see <https://github.com/microsoft/Deep3DFaceReconstruction>

```bash
pip install tensorflow-gpu
```

### Install Python Libraries

```bash
pip install dlib --verbose
pip intsll numpy
pip install Pillow
pip install scikit-image
pip install scipy
pip install imutils
pip install opencv-python
```

## Run

```bash
python app.py
```
