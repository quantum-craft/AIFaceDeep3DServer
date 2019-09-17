FROM tensorflow/tensorflow:latest-gpu-py3

LABEL maintainer "hallo.quantum@gmail.com"

COPY 01_MorphableModel.mat /home/ubuntu/
COPY BFM_model_front.mat /home/ubuntu/
COPY Exp_Pca.bin /home/ubuntu/
COPY FaceReconModel.pb /home/ubuntu/

RUN apt-get update && apt-get install -y \
git \
libsm6 \
libxext6 \
libxrender-dev \
net-tools

RUN git clone --recursive https://github.com/quantum-craft/AIFaceDeep3DServer.git

RUN pip install --upgrade pip
RUN pip install cmake
RUN pip install dlib
RUN pip install numpy \
Pillow \
scikit-image \
scipy \
imutils \
opencv-python \
flask-restful

RUN cp /home/ubuntu/FaceReconModel.pb /AIFaceDeep3DServer/algorithms/Deep3DFaceReconstruction/network/
RUN cp /home/ubuntu/01_MorphableModel.mat /AIFaceDeep3DServer/algorithms/Deep3DFaceReconstruction/BFM/
RUN cp /home/ubuntu/BFM_model_front.mat /AIFaceDeep3DServer/algorithms/Deep3DFaceReconstruction/BFM/
RUN cp /home/ubuntu/Exp_Pca.bin /AIFaceDeep3DServer/algorithms/Deep3DFaceReconstruction/BFM/

WORKDIR /AIFaceDeep3DServer
ENV ip 0.0.0.0
ENV port 5000

CMD ["sh", "-c", "python app.py --ip ${ip} --port ${port}"]

