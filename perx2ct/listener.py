import io
from multiprocessing.connection import Listener

import numpy as np
import torch

from inference import Inference

CHECKPOINT_PATH = '/home/sebastianp/code/uni/appliedml/X2CT-MMe3D/perx2ct/PerX2CT/checkpoints/PerX2CT.ckpt'
CONFIG_PATH = '/home/sebastianp/code/uni/appliedml/X2CT-MMe3D/perx2ct/PerX2CT/configs/PerX2CT.yaml'
CT_EXTENSION = 'h5'
DEVICE = torch.device('cpu')

def main():
    listener = Listener(('localhost', 6000))
    model = Inference(CONFIG_PATH, CHECKPOINT_PATH, DEVICE)

    conn = listener.accept()

    print("Connected!", conn)

    while True:
        try:
            data = conn.recv()
        except EOFError:
            print('Connection closed by server')

            break

            continue

        if data is None:
            break

        data = conn.recv()
        tensor = np.load(io.BytesIO(data))

        print('process b receive', tensor)

        result = tensor + 1

        print('process b result', result)

        buf = io.BytesIO()
        np.save(buf, result)

        continue

        images = np.load(io.BytesIO(data))
        volume = model(images[0], images[1])

        buf = io.BytesIO()
        np.save(buf, volume.cpu().numpy(), allow_pickle=False)

        conn.send(buf.getvalue())




main()