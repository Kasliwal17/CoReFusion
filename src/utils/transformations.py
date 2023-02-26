import albumentations as albu

def get_training_augmentation():
    train_transform = [
        albu.Resize(480,640,always_apply=True),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),

#         albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
        
#         albu.RandomCrop(height=512, width=512),
#         albu.ColorJitter(p=0.5),
#         albu.IAAAdditiveGaussianNoise(p=0.2),
#         albu.IAAPerspective(p=0.5),
        albu.PadIfNeeded(min_height=480, min_width=640, always_apply=True, border_mode=0),
#         albu.OneOf(
#             [
#                 albu.CLAHE(p=1),
#                 albu.RandomBrightness(p=1),
#                 albu.RandomGamma(p=1),
#             ],
#             p=0.9,
#         ),

#         albu.OneOf(
#             [
#                 albu.IAASharpen(p=1),
#                 albu.Blur(blur_limit=3, p=1),
#                 albu.MotionBlur(blur_limit=3, p=1),
#             ],
#             p=0.9,
#         ),

#         albu.OneOf(
#             [
#                 albu.RandomContrast(p=1),
#                 albu.HueSaturationValue(p=1),
#             ],
#             p=0.9,
#         ),
    ]
    return albu.Compose(train_transform,additional_targets={'image1':'mask'})

def get_pretraining_augmentation():
    train_transform = [
        albu.Resize(480,640,always_apply=True),
        albu.HorizontalFlip(p=0.4),
        albu.VerticalFlip(p=0.4),

        # albu.ShiftScaleRotate(p=0.4, border_mode=0),
        
        albu.RandomCrop(height=400, width=580, p=0.4),
#         albu.ColorJitter(p=0.5),
        albu.IAAAdditiveGaussianNoise(p=0.4),
#         albu.IAAPerspective(p=0.5),
        albu.Resize(480,640,always_apply=True),
        albu.PadIfNeeded(min_height=480, min_width=640, always_apply=True, border_mode=0),
#         albu.OneOf(
#             [
#                 albu.CLAHE(p=1),
#                 albu.RandomBrightness(p=1),
#                 albu.RandomGamma(p=1),
#             ],
#             p=0.9,
#         ),

#         albu.OneOf(
#             [
#                 albu.IAASharpen(p=1),
#                 albu.Blur(blur_limit=3, p=1),
#                 albu.MotionBlur(blur_limit=3, p=1),
#             ],
#             p=0.9,
#         ),

#         albu.OneOf(
#             [
#                 albu.RandomContrast(p=1),
#                 albu.HueSaturationValue(p=1),
#             ],
#             p=0.9,
#         ),
    ]
    return albu.Compose(train_transform,additional_targets={'image1':'mask'})
    
def get_validation_augmentation():
    test_transform = [
        albu.Resize(480,640,always_apply=True),
        albu.PadIfNeeded(480,640)
    ]
    return albu.Compose(test_transform,additional_targets={'image1':'mask'})


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)
    