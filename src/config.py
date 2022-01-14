import torch


IMAGES_DIR = '../Animals_with_Attributes2/JPEGImages/'
LABEL_FILE = '../Animals_with_Attributes2/classes.txt'
TRAIN_CAP = 8000
TEST_CAP = 2000

IMAGE_SIZE = 32
PATCH_SIZE = 4
NUM_CLASSES = 50
DIM = 768
HEADS = 8
DEPTH = 6
MLP_DIM = 1024

NORMALIZE_MEAN = 0.0
NORMALIZE_STD = 1.0
MAX_PIXEL_VALUE = 255.0

BATCH_SIZE = 64
EPOCHS = 20
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'