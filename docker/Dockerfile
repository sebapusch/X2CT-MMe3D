FROM debian:bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    git \
    bzip2 \
    wget \
    build-essential \
    cmake \
    pkg-config \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set up conda path
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Install conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    mkdir -p $CONDA_DIR && \
    bash /tmp/miniconda.sh -b -u -p $CONDA_DIR && \
    rm /tmp/miniconda.sh

# Copy source
COPY .. /app
WORKDIR /app

# Set up PerX2CT
RUN cd ./perx2ct/scripts && chmod +x setup.sh && ./setup.sh

# Prepare perx2ct python env
RUN conda create -y -n perx2ct python=3.8
RUN conda run --no-capture-output -n perx2ct pip install -v --upgrade pip && \
    conda run --no-capture-output -n perx2ct pip install -v \
      torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 \
      -f https://download.pytorch.org/whl/torch_stable.html && \
    conda run --no-capture-output -n perx2ct pip install -v -r /app/perx2ct/PerX2CT/requirement.txt

# Prepare app python env
RUN conda create -y -n py310 python=3.10
RUN conda run --no-capture-output -n py310 pip install -v --upgrade pip && \
    conda run --no-capture-output -n py310 pip install -v -r /app/requirements.txt

# Set api workdir
WORKDIR /app/api

# Add app root to python path
ENV PYTHONPATH="/app"

# Add entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]