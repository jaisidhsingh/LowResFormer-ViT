TRAIN_FILES = {
    'attr': "../../deepfashion/coarse/list_attr_img.txt",
    'cats': "../../deepfashion/coarse/list_category_img.txt",
}

TEST_FILES = {
    'attr': "../../deepfashion/coarse/list_attr_img.txt",
    'cats': "../../deepfashion/coarse/list_category_img.txt",
}

TRAIN_DIR = "../../deepfashion/coarse/train/"
VAL_DIR = "../../deepfashion/coarse/val/"
TEST_DIR = "../../deepfashion/coarse/test/"

IMAGE_FOLDER = "../../deepfashion/img_highres/"

LOG_FILE = "./logs.json"
CKPT_DIR = "./checkpoints/"

IMAGE_FOLDER = "../../deepfashion/img_highres"

TRAIN_CAP = [0, 209219]
VAL_CAP = [209219, 249219]
TEST_CAP = [249219, 289219]

NORMALIZE_MEAN = 0.0
NORMALIZE_STD = 1.0
MAX_PIXEL_VALUE = 255.0

IN_CHANNELS = 3
IMAGE_SIZE = 32
PATCH_SIZE = 4
INT_DIM = (IMAGE_SIZE // PATCH_SIZE)**2

NUM_CLASSES = 50
NUM_ATTR = 26
ATTR_EMB = 64
DIM = 768
HEADS = 8
DEPTH = 6
MLP_DIM = 1024
EXPANSION_FACTOR = 4
DROPOUT = 0.1

NORMALIZE_MEAN = 0.0
NORMALIZE_STD = 1.0
MAX_PIXEL_VALUE = 255.0

BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 1e-6
DEVICE = 'cuda'

ALPHA = 1
BETA = 0
GAMMA = 0
