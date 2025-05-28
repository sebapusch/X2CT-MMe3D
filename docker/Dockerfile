FROM debian:bookworm

# System dependencies
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

# Install Miniconda
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    mkdir -p $CONDA_DIR && \
    bash /tmp/miniconda.sh -b -u -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    conda init bash

# Copy source
COPY .. /app
WORKDIR /app

# Create conda env for main app (Python 3.10)
RUN conda create -y -n py310 python=3.10 && \
    conda run -n py310 pip install -r requirements.txt

# Create perx2ct env with Python 3.8 and install dependencies
RUN conda create -y -n perx2ct python=3.8 && \
    conda run -n perx2ct pip install --upgrade pip && \
    conda run -n perx2ct pip install \
      torch==1.8.1+cu111 \
      torchvision==0.9.1+cu111 \
      torchaudio==0.8.1 \
      -f https://download.pytorch.org/whl/torch_stable.html && \
    conda run -n perx2ct pip install -r /app/perx2ct/PerX2CT/requirement.txt

# Run setup script (installs patches, config, checkpoint)
RUN cd /app/scripts && chmod +x setup.sh && ./setup.sh

WORKDIR /app/api

# Use Python 3.10 app, pass path to Python 3.8 env for listener
CMD ["conda", "run", "-n", "py310", "python", "main.py", "--python_path", "/opt/conda/envs/perx2ct/bin/python"]
