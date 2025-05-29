import io
import logging
import subprocess
import time
from multiprocessing.connection import Client

import numpy as np


class PerX2CTClient:
    def __init__(self,
                 python_path: str,
                 script_path: str,
                 model_path: str,
                 config_path: str,
                 port: int = 6000):
        self.config = {
            'python_path': python_path,
            'script_path': script_path,
            'model_path': model_path,
            'config_path': config_path,
        }

        self.port = port
        self.process = None
        self.client = None

    def start_process(self):
        logging.info("Starting PerX2CT subprocess...")
        self.process = subprocess.Popen([
            self.config['python_path'],
            self.config['script_path'],
            '--checkpoint', self.config['model_path'],
            '--config', self.config['config_path'],
            '--port', str(self.port),
        ])

        for _ in range(50):
            try:
                self.client = Client(('localhost', self.port))
                logging.info("PerX2CT connection established")
                return
            except Exception:
                time.sleep(0.2)
        raise RuntimeError("Could not connect to listener")

    def stop_process(self):
        logging.info("Shutting down listener subprocess...")
        if self.process:
            self.process.terminate()

    def send(self, data: np.ndarray):
        buf = io.BytesIO()
        np.save(buf, data, allow_pickle=False)
        self.client.send(buf.getvalue())

    def receive(self) -> np.ndarray:
        data = self.client.recv()
        np_data = np.load(io.BytesIO(data), allow_pickle=False)
        if np_data.shape == (0,):
            raise ValueError('Received error message')

        return np_data