# ! USER SPECIFIC
# <<<<<<<<<<<<<<<<<<<<<<<<<
# Edit the base image here, e.g., to use 
# TENSORFLOW (https://hub.docker.com/r/tensorflow/tensorflow/) 
# or a different PYTORCH (https://hub.docker.com/r/pytorch/pytorch/) base image
FROM nvidia/cuda:11.3.1-runtime-ubuntu16.04 AS sati-base


# Install Miniconda in /opt/conda
#

ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# setup conda virtual environment
COPY requirements.yml /tmp/requirements.yml
RUN conda update conda \
    && conda env create -f /tmp/requirements.yml

# activate conda env
RUN echo "conda activate StitchingAwareTrainingAndInference" >> ~/.bashrc
ENV PATH /opt/conda/envs/StitchingAwareTrainingAndInference/bin:$PATH
ENV CONDA_DEFAULT_ENV $StitchingAwareTrainingAndInference

# add group:user algorithm:algorithm and activate user
RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm
USER algorithm

# copy files
COPY *.py ./
COPY eval eval
COPY benchmark_dataset benchmark_dataset

# script to execute during docker run
CMD ["sh", "./run.sh"]