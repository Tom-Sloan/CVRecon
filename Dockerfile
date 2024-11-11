FROM nvidia/cuda:12.4-cudnn8-runtime-ubuntu20.04

# Set the working directory
WORKDIR /app

# Install Miniconda
RUN apt-get update && apt-get install -y wget bzip2 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Add conda to PATH
ENV PATH=/opt/conda/bin:$PATH

# Create conda environment and install packages
RUN conda create -n cvrecon python=3.11 -y && \
    conda init bash && \
    /bin/bash -c "source activate cvrecon && \
    conda config --add channels conda-forge && \
    conda install -c conda-forge pytorch-lightning && \
    conda install -c conda-forge scikit-image pillow opencv-python imageio && \
    conda install -c conda-forge open3d pyrender trimesh && \
    conda install -c conda-forge wandb tqdm ray pyyaml matplotlib black numba && \
    conda install -c conda-forge pycuda"

# Install torchsparse
RUN apt-get install -y libsparsehash-dev && \
    git clone git@github.com:mit-han-lab/torchsparse.git && \
    cd torchsparse && \
    pip install -r requirements.txt && \
    pip install -e . && \
    cd .. && \
    pip install -e .
