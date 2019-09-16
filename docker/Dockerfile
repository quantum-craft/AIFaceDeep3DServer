FROM tensorflow/tensorflow:latest-gpu-py3

LABEL maintainer "hallo.quantum@gmail.com"

COPY 01_MorphableModel.mat /home/ubuntu/
COPY BFM_model_front.mat /home/ubuntu/
COPY Exp_Pca.bin /home/ubuntu/
COPY FaceReconModel.pb /home/ubuntu/

RUN apt-get update

RUN apt-get install -y git

RUN apt-get install -y libsm6 libxext6 libxrender-dev

RUN apt-get install -y net-tools

RUN git clone --recursive https://github.com/quantum-craft/AIFaceDeep3DServer.git

RUN pip install --upgrade pip

RUN pip install cmake

RUN pip install dlib --verbose

RUN pip install numpy

RUN pip install Pillow

RUN pip install scikit-image

RUN pip install scipy

RUN pip install imutils

RUN pip install opencv-python

RUN pip install flask-restful

RUN cp /home/ubuntu/FaceReconModel.pb /AIFaceDeep3DServer/algorithms/Deep3DFaceReconstruction/network/
RUN cp /home/ubuntu/01_MorphableModel.mat /AIFaceDeep3DServer/algorithms/Deep3DFaceReconstruction/BFM/
RUN cp /home/ubuntu/BFM_model_front.mat /AIFaceDeep3DServer/algorithms/Deep3DFaceReconstruction/BFM/
RUN cp /home/ubuntu/Exp_Pca.bin /AIFaceDeep3DServer/algorithms/Deep3DFaceReconstruction/BFM/
