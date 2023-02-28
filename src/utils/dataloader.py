from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import cv2
from .misc import normalize_data, list_img, preprocessing_fn
from .transformations import to_tensor

class Dataset(BaseDataset):  
    def __init__(
            self, 
            hr_dir: str, 
            thermal_dir:str,
            tar_dir: str, 
            augmentation=None, 
    ):
        self.hr_list = list_img(hr_dir)
        self.thermal_list= list_img(thermal_dir)
        self.tar_list = list_img(tar_dir)
        self.augmentation = augmentation
        self.preprocessing = preprocessing_fn
        self.to_tensor = to_tensor
    
    def __getitem__(self, i):
        
        # read data
        himage = cv2.imread(self.hr_list[i], 0)
#         himage = cv2.cvtColor(himage, cv2.COLOR_BGR2RGB)
        target = cv2.imread(self.tar_list[i], 0)
        timage = cv2.imread(self.thermal_list[i], 0)
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=himage,image1=timage, mask=target)
            himage, target, timage= sample['image'], sample['mask'], sample['image1']
        target = target.reshape(480,640,1)
        timage = timage.reshape(480,640,1)
        himage = himage.reshape(480,640,1)
        if self.preprocessing:
            himage = self.to_tensor(self.preprocessing(himage))
            timage = self.to_tensor(self.preprocessing(timage))
            target = self.to_tensor(target)
            target = target/255
            target = normalize_data(target)
        return himage,timage, target
        
    def __len__(self):
        return len(self.hr_list)
