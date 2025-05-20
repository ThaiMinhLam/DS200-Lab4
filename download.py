import argparse
import kagglehub
import os
import shutil

def download_dataset(path_dataset, path_destination):
    source_path = kagglehub.dataset_download(path_dataset)
    os.makedirs(path_destination, exist_ok=True)
    
    for item in os.listdir(source_path):
        src_item = os.path.join(source_path, item)
        dst_item = os.path.join(path_destination, item)
        shutil.move(src_item, dst_item)
    print(f"Đã di chuyển toàn bộ dữ liệu đến: {path_destination}")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download dataset')
    parser.add_argument("--path_dataset", help="Kaggle Dataset", type=str, required=True)
    parser.add_argument("--path_destination", help="Direction Save Dataset", type=str, required=True)
    
    args = parser.parse_args()
    download_dataset(path_dataset=args.path_dataset, path_destination=args.path_destination)