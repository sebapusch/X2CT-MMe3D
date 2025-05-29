import argparse
import io
import logging
from multiprocessing.connection import Listener, Connection

import numpy as np
import torch

from inference import Inference

DEVICE = torch.device('cpu')

class ArrayTransferConnection:
    def __init__(self, conn: Connection):
        self.conn = conn

    def send(self, data: np.ndarray):
        buf = io.BytesIO()
        np.save(buf, data, allow_pickle=False)
        self.conn.send(buf.getvalue())

    def receive(self) -> np.ndarray:
        data = self.conn.recv()
        return np.load(io.BytesIO(data), allow_pickle=False)

    def error(self):
        self.send(np.empty((0,)))

    def close(self):
        self.conn.close()


def main(args: argparse.Namespace):
    logging.info(f'Starting listener on port {args.port}...')
    listener = Listener(('localhost', args.port))
    model = Inference(args.config, args.checkpoint, DEVICE)

    connection = ArrayTransferConnection(
        listener.accept())
    logging.info("Connected to client")

    while True:
        try:
            frontal = connection.receive()
            lateral = connection.receive()

            volume = model(frontal, lateral)

            connection.send(volume.cpu().numpy())

        except EOFError:
            logging.info("Listener closed")
            break

        except Exception as e:
            logging.error(f'Unexpected error during inference: {e}')
            connection.error()
            continue

    connection.close()
    logging.info("Shut down listener")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="[perx2ct] [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="CT volume listener for PerX2CT")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--port", type=int, default=6000)

    main(parser.parse_args())
