# create image list from test/train.json

import json
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('json', help='input json file path')
    parser.add_argument('-o', '--out', default='img_list.list', help='output list file path')
    return parser.parse_args()

def main():
    args = parse_args()
    json_path = args.json
    json_open = open(json_path, 'r')
    json_load = json.load(json_open)

    dir_path =os.path.dirname(json_path)
    img_list_name = args.out

    for im in json_load['images']:
        img_path = os.path.join(dir_path, im['file_name'])
        print(img_path)

        with open(img_list_name, mode='a') as f:
            f.writelines(img_path+'\n')

    print('Exprot: {}'.format(img_list_name))

if __name__ == '__main__':
    main()