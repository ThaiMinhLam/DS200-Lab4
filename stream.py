import time
import json
import pickle
import socket
import argparse
import numpy as np
from tqdm import tqdm
import os

from torch.utils.data import DataLoader
from dataset import ImageDataset

parser = argparse.ArgumentParser(description='Streams a file to a Spark Streaming Context')
parser.add_argument('--folder', '-f', help='Data folder', required=True, type=str)
parser.add_argument('--batch_size', '-b', help='Batch size', required=True, type=int)
parser.add_argument('--split','-s', help="training or test split", required=False, type=str, default='train')
parser.add_argument('--sleep','-t', help="streaming interval", required=False, type=int, default=3)
args = parser.parse_args()

TCP_IP = "localhost"
TCP_PORT = 6100

class Streamer:
    def __init__(self, dataset, batch_size, split, sleep_time):
        self.batch_size = batch_size
        self.split = split
        self.sleep_time = sleep_time
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size, 
            shuffle=True,
            num_workers=2
        )
        
    def connect_TCP(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((TCP_IP, TCP_PORT))
        s.listen(1)
        print(f"[Streamer] Waiting for connection on {TCP_IP}: {TCP_PORT} …")
        connection, address = s.accept()
        print(f"[Streamer] Connected to {address}")
        self.connection = connection
        return connection, address
    
    def send_batch(self, images, labels):
        flat_images = images.view(images.size(0), -1).tolist()
        labels = labels.tolist()
        
        payload = {
            i: {**{f'feature-{j}': value for j, value in enumerate(features)}, 'label': label}
            for i, (features, label) in enumerate(zip(flat_images, labels))
        }
        
        message = (json.dumps(payload) + '\n').encode('utf-8')
        
        try:
            self.connection.sendall(message)
        except BrokenPipeError:
            raise ValueError("Connection lost.")

        except Exception as e:
            raise ValueError(f"Unexpected error: {e}")

        return True
    
    def stream_dataset(self):
        with tqdm(total=len(self.loader), desc="Streaming Batches") as pbar:
            for imgs, labels in self.loader:
                if not self.send_batch(imgs, labels):
                    break

                pbar.update(1)
                time.sleep(self.sleep_time)
                
if __name__ == '__main__':
    dataset = ImageDataset(
        directory=os.path.join(args.folder, args.split),
    )
    streamer = Streamer(dataset, args.batch_size, args.split, args.sleep)
    streamer.connect_TCP()

    if args.endless == True:
        while True:
            streamer.stream_dataset()
    else:
        streamer.stream_dataset()

    streamer.conn.close()