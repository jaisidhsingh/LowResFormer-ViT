CUB_FILES = {
	'dataset_dir': '../CUB/images/',
	'image_file': '../CUB/lists/files.txt',
	'label_file' : '../CUB/lists/classes.txt',
	'attribute_file' : '../CUB/attributes/labels.txt',
	'image_keys' : '../CUB/attributes/images-dirs.txt'
}

AWA_TRAIN_CAP = 25900
AWA_TEST_CAP = 11100

CUB_TRAIN_CAP = 5000
CUB_TEST_CAP = 1000

IN_CHANNELS = 3
IMAGE_SIZE = 16
PATCH_SIZE = 2
INT_DIM = (IMAGE_SIZE // PATCH_SIZE)**2

NUM_CLASSES = 50
NUM_ATTR = 85

CUB_CLASSES = 200
CUB_ATTR = 312

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

BATCH_SIZE = 1
EPOCHS = 20
LEARNING_RATE = 1e-6
DEVICE = 'cuda'

ALPHA = 1
BETA = 0
GAMMA = 0