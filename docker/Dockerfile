FROM debian:bookworm

# Install dependencies.
RUN apt-get update
RUN apt-get install  -y --no-install-recommends \
    ca-certificates \
    gnupg \
    dirmngr \
    curl \
    git

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
RUN python3.10 -m pip install --break-system-packages -r requirements.txt \
    && python3.10 -m pip install --break-system-packages \
      torch==2.7.0+cpu \
      torchvision==0.22.0+cpu \
      -f https://download.pytorch.org/whl/torch_stable.html


# Setup perx2ct
RUN cd ./perx2ct/scripts \
    && chmod +x ./setup.sh \
    && ./setup.sh

# Install perx2ct dependencies
RUN cd ./perx2ct/PerX2CT \
    && python3.8 -m pip install --break-system-packages --upgrade pip \
    && python3.8 -m pip install --break-system-packages \
      torch==1.8.1+cu111 \
      torchvision==0.9.1+cu111 \
      torchaudio==0.8.1  \
      -f https://download.pytorch.org/whl/torch_stable.html  \
    && python3.8 install --break-system-packages -r requirement.txt

WORKDIR /app/api

CMD ["python3.10", "--python_path", "python3.8"]