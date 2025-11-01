import os
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--input_json_path', type=str, required=True, help='Path to the input JSON path')
parser.add_argument('--data_path', type=str, required=True, help='Path to the image folder')
args = parser.parse_args()


if __name__ == '__main__':
    input_path = args.input_json_path
    data_path = args.data_path

    with open(input_path, 'r') as f:
        data = json.load(f)

    save_data = {}
    _keys = list(data.keys())
    _keys.sort()
    for key in _keys:
        _data = data[key]
        save_item = {}
        save_item['tool_returned_web_title'] = _data['tool_returned_web_title']
        cached_images_path = []
        _cached_images_path = _data['cached_images_path']
        for img_path in _cached_images_path:
            if not img_path:
                cached_images_path.append(None)
                continue
            new_path = os.path.join(data_path, img_path)
            cached_images_path.append(new_path)
        save_item['cached_images_path'] = cached_images_path
        save_data[key] = save_item

    with open(input_path, 'w') as f:
        json.dump(save_data, f, indent=4, ensure_ascii=False)







