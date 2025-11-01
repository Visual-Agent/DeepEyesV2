import os
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, required=True, help='Path to the input JSON path')
parser.add_argument('--data_path', type=str, required=True, help='Path to the image folder')
args = parser.parse_args()


if __name__ == '__main__':
    input_path = args.input_path
    data_path = args.data_path

    json_files = [f for f in os.listdir(input_path) if f.endswith('.json')]

    for json_file in json_files:
        with open(os.path.join(input_path, json_file), 'r') as f:
            data = json.load(f)
        save_data = []
        for item in data:
            if 'images' in item and len(item['images']) > 0:
                images = [os.path.join(data_path, img) for img in item['images']]
                item['images'] = images
            save_data.append(item)
        with open(os.path.join(input_path, json_file), 'w') as f:
            json.dump(save_data, f, indent=4)
        print(f'Processed {json_file}, total items: {len(save_data)}')







