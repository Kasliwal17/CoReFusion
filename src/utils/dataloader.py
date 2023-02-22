from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import cv2
from .misc import normalize_data

class Dataset(BaseDataset):  
    def __init__(
            self, 
            images_dir: list, 
            fc_dir:list,
            masks_dir: list, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.images_list = images_dir
        self.fc_list=fc_dir
        self.masks_list = masks_dir
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        himage = cv2.imread(self.images_list[i])
        himage = cv2.cvtColor(himage, cv2.COLOR_BGR2RGB)
        target = cv2.imread(self.masks_list[i], 0)
        timage = cv2.imread(self.fc_list[i])
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=himage,image1=timage, mask=target)
            himage, target, timage= sample['image'], sample['mask'], sample['image1']
        target = target.reshape(480,640,1)
#         timage = timage.reshape(480,640,1)
        if self.preprocessing:
            sample = self.preprocessing(image=himage, mask=target)
            himage, target = sample['image'], sample['mask']
            sample = self.preprocessing(image=timage)
            timage= sample['image']
            sample = self.preprocessing
            target = target/255
            target = normalize_data(target)
        return himage,timage, target#, label
        
    def __len__(self):
        return len(self.images_list)
