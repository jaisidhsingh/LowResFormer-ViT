import torch


IMAGES_DIR = '../Animals_with_Attributes2/JPEGImages/'
LABEL_FILE = '../Animals_with_Attributes2/classes.txt'

IMAGE_SIZE = 224
PATCH_SIZE = 14
NUM_CLASSES = 50
DIM = 768
HEADS = 8
DEPTH = 6
MLP_DIM = 1024

NORMALIZE_MEAN = 0.0
NORMALIZE_STD = 1.0
MAX_PIXEL_VALUE = 255.0

BATCH_SIZE = 32
EPOCHS = 10
DEVICE = 'cuda'