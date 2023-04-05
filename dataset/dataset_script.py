import os
from os import path
import requests
import numpy as np
import pandas as pd
from datasets import load_dataset
from PIL import Image
import time
start = time.time()

OUTPUT_DIR = 'H:\Downloads\School\Year5\comp4107\project\\test\\test_dataset'
TRAIN_PATH = path.join(OUTPUT_DIR, 'train')
VAL_PATH = path.join(OUTPUT_DIR, 'validation')
TEST_PATH = path.join(OUTPUT_DIR, 'test')

'''TRAIN_SIZE = 3
VAL_SIZE = 2
TEST_SIZE = 5'''
TRAIN_SIZE = 500
VAL_SIZE = 100
TEST_SIZE = 600

'''
Output File Structure
--------------
OUTPUT_DIR
  - train
    - non_ai
      - non_ai_1.png
      - ...
    - ai
      - ai_1.png
      - ...
  - test
    - non_ai
      - non_ai_1.png
      - ...
    - ai
      - ai_1.png
      - ...
  - validation
    - non_ai
      - non_ai_1.png
      - ...
    - ai
      - ai_1.png
      - ...
'''

def getImageExtension(format):
    if format.upper() == 'PNG':
        return '.png'
    return '.jpg'


def loadDataNonAI(dataset, take, output_path, filename):
    print("Loading to", output_path)
    i = 0
    rows = dataset.take(take)
    for row in rows:
        print("Loading non-AI image:", i)
        try:
            img = Image.open(requests.get(row['URL'], stream=True).raw)
            full_path = path.join(output_path, filename + '_' + str(i) + getImageExtension(img.format))
            img.save(full_path)
        except Exception as e:
            print("image at", str(i), "failed to load/save")
            print(e)
        i += 1


def loadDataAI(dataset, take, output_path, filename):
    print("Loading to", output_path)
    i = 0
    rows = dataset.take(take)
    for row in rows:
        print("Loading AI image:", i)
        try:
            img = row['image']
            full_path = path.join(output_path, filename + '_' + str(i) + getImageExtension(img.format))
            img.save(full_path)
        except Exception as e:
            print("image at", str(i), "failed to load/save")
            print(e)
        i += 1


os.makedirs(path.join(TRAIN_PATH, 'non_ai'), exist_ok=True)
os.makedirs(path.join(TRAIN_PATH, 'ai'), exist_ok=True)
os.makedirs(path.join(VAL_PATH, 'non_ai'), exist_ok=True)
os.makedirs(path.join(VAL_PATH, 'ai'), exist_ok=True)
os.makedirs(path.join(TEST_PATH, 'non_ai'), exist_ok=True)
os.makedirs(path.join(TEST_PATH, 'ai'), exist_ok=True)


# non-ai art

dataset = load_dataset('laion/laion-art', split='train', streaming=True)
loadDataNonAI(dataset, TRAIN_SIZE, path.join(TRAIN_PATH, 'non_ai'), 'non_ai')
loadDataNonAI(dataset.skip(TRAIN_SIZE), VAL_SIZE, path.join(VAL_PATH, 'non_ai'), 'non_ai')
loadDataNonAI(dataset.skip(TRAIN_SIZE+VAL_SIZE), TEST_SIZE, path.join(TEST_PATH, 'non_ai'), 'non_ai')

# stable diffusion
# load with `2m_first_50k` or `2m_first_5k` subset
dataset = load_dataset('poloclub/diffusiondb', '2m_first_5k', split='train', streaming=True)
loadDataAI(dataset, TRAIN_SIZE, path.join(TRAIN_PATH, 'ai'), 'ai')
loadDataAI(dataset.skip(TRAIN_SIZE), VAL_SIZE, path.join(VAL_PATH, 'ai'), 'ai')
loadDataAI(dataset.skip(TRAIN_SIZE+VAL_SIZE), TEST_SIZE, path.join(TEST_PATH, 'ai'), 'ai')


# download midjourney?


# download DALL-E?


print('Took', time.time()-start, 'seconds.')