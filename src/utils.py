import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import *

train_transforms = A.Compose(
	[
		A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
		A.Normalize(
         mean=[NORMALIZE_MEAN for _ in range(3)],
         std=[NORMALIZE_STD for _ in range(3)],
      	max_pixel_value=MAX_PIXEL_VALUE,
      ),
		ToTensorV2()
	]
)

test_transforms = A.Compose(
	[
		A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
		A.Normalize(
         mean=[NORMALIZE_MEAN for _ in range(3)],
         std=[NORMALIZE_STD for _ in range(3)],
      	max_pixel_value=MAX_PIXEL_VALUE,
      ),
		ToTensorV2()
	]
)