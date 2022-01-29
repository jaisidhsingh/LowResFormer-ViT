import torch


IMAGES_DIR = '../Animals_with_Attributes2/JPEGImages/'
LABEL_FILE = '../Animals_with_Attributes2/classes.txt'
ATTRIBUTE_FILE = '../Animals_with_Attributes2/predicate-matrix-binary.txt'

TRAIN_CAP = 8000
TEST_CAP = 500
IN_CHANNELS = 3
IMAGE_SIZE = 32
PATCH_SIZE = 4
INT_DIM = (IMAGE_SIZE // PATCH_SIZE)**2 + 1

NUM_CLASSES = 50
NUM_ATTR = 85
ATTR_EMB = 64
DIM = 768
HEADS = 8
DEPTH = 6
MLP_DIM = 1024
EXPANSION_FACTOR = 4
DROPOUT = 0.0

NORMALIZE_MEAN = 0.0
NORMALIZE_STD = 1.0
MAX_PIXEL_VALUE = 255.0

BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'