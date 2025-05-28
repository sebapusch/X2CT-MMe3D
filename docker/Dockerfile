FROM debian:bookworm

# Install dependencies.
RUN apt-get update
RUN apt-get install  -y --no-install-recommends \
    ca-certificates \
    gnupg \
    dirmngr \
    curl \
    git \
    build-essential \
    cmake \
    pkg-config \
    libgl1 \
    libglib2.0-0

RUN apt-get clean autoclean \
  && apt-get autoremove -y \
  && rm -rf /var/lib/apt/lists/* \
  && rm -f /var/cache/apt/archives/*.deb

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh

# Install python versions
RUN uv python install 3.10 3.8

# Link python versions
RUN bash -c 'ln -s $(uv python find 3.10) /usr/local/bin/python3.10'
RUN bash -c 'ln -s $(uv python find 3.8)  /usr/local/bin/python3.8'

COPY .. /app
WORKDIR /app

# Install pip dependencies
RUN python3.10 -m pip install --break-system-packages --upgrade pip \
    && python3.10 -m pip install --break-system-packages -r requirements.txt


# Setup perx2ct
RUN cd ./perx2ct/scripts \
    && chmod +x ./setup.sh \
    && ./setup.sh

# Install perx2ct dependencies
RUN cd ./perx2ct/PerX2CT \
    && python3.8 -m pip install --break-system-packages --upgrade pip \
    && python3.8 -m pip install --break-system-packages numpy \
    && python3.8 -m pip install --break-system-packages \
      torch==2.0.0 \
      torchvision==0.15.1 \
      torchaudio==2.0.1 \
    && python3.8 -m pip install --break-system-packages --no-deps elasticdeform \
    && python3.8 -m pip install --break-system-packages -r requirement.txt

WORKDIR /app/api

# ENV PYTHONPATH="/app"

# python3.10 main.py --python_path python3.8docker run -it -p 8000:8000 med:latest sh

CMD ["python3.10", "main.py", "--python_path", "python3.8"]