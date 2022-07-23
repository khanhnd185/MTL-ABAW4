import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def get_dataset(filename, feature_dict, onlyfilename=False):
    with open(filename) as f:
        lines = f.read().splitlines()
    num_missed = 0
    X, y_va, y_ex, y_au = [], [], [], []
    masks_va, masks_ex, masks_au = [], [], []

    for line in lines[1:]:
        splitted_line = line.split(',')
        imagename = splitted_line[0]
        valence = float(splitted_line[1])
        arousal = float(splitted_line[2])
        expression = int(splitted_line[3])
        aus = list(map(int, splitted_line[4:]))
        
        mask_va = (valence > -5 and arousal > -5)
        if not mask_va:
            valence = arousal = 0
        mask_ex = (expression > -1)
        if not mask_ex:
            expression = 0
        mask_au = min(aus) >= 0
        if not mask_au:
            aus = [0]*len(aus)

        if mask_va or mask_ex or mask_au:
            if imagename in feature_dict or onlyfilename:
                if onlyfilename:
                    X.append(imagename)
                else:
                    X.append(np.concatenate((feature_dict[imagename][0], feature_dict[imagename][1])))
                y_va.append((valence, arousal))
                masks_va.append(mask_va)
                
                y_ex.append(expression)
                masks_ex.append(mask_ex)
                
                y_au.append(aus)
                masks_au.append(mask_au)
            else:
                num_missed += 1

    X = np.array(X)
    y_va = np.array(y_va)
    y_ex = np.array(y_ex)
    y_au = np.array(y_au)
    masks_va = np.array(masks_va).astype(np.float32)
    masks_ex = np.array(masks_ex).astype(np.float32)
    masks_au = np.array(masks_au).astype(np.float32)

    return X, y_va, y_ex, y_au, masks_va, masks_ex, masks_au

class ImageSAW2(Dataset):
    def __init__(self, filename, img_path):
        super(ImageSAW2, self).__init__()
        with open(filename) as f:
            lines = f.read().splitlines()
        self.X = lines[1:]
        self.img_path = img_path
        self.transform = transforms.Compose(
            [
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            ]
        )
    
    def __getitem__(self, i):
        img = pil_loader(os.path.join(self.img_path, self.X[i]))
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.X)

class UniSAW2(Dataset):
    def __init__(self, filename, feature_dict, task='All'):
        super(UniSAW2, self).__init__()
        y = {}
        mask = {}
        self.y = {}
        self.task = task
        X , y['VA'], y['EX'], y['AU'], mask['VA'], mask['EX'], mask['AU'] = get_dataset(filename, feature_dict)

        if task == 'All':
            self.X = X
            self.y = y
            self.mask = mask
        else:
            self.X = X[mask[task] == 1]
            self.y[task] = y[task][mask[task] == 1]

        print(len(self.X))

    def __getitem__(self, i):
        if self.task == 'All':
            return self.X[i] , self.y['VA'][i], self.y['EX'][i], self.y['AU'][i], self.mask['VA'][i], self.mask['EX'][i], self.mask['AU'][i]
        else:
            return self.X[i] , self.y[self.task][i]

    def __len__(self):
        return len(self.X)

    def ex_weight(self):
        if self.task == 'EX':
            return compute_class_weight(class_weight='balanced', classes=np.unique(self.y['EX'].astype(int)), y=self.y['EX'].astype(int))
        else:
            return np.zeros(8)

    def au_weight(self):
        if self.task == 'AU':
            weight = 1.0 / np.sum(self.y['AU'], axis=0)
            weight = weight / weight.sum() * self.y['AU'].shape[1]
            return weight
        else:
            return np.zeros(12)

class RawSAW2(Dataset):
    def __init__(self, filename, img_path, task='All'):
        super(RawSAW2, self).__init__()
        y = {}
        mask = {}
        self.y = {}
        self.task = task
        self.img_path = img_path
        X , y['VA'], y['EX'], y['AU'], mask['VA'], mask['EX'], mask['AU'] = get_dataset(filename, {}, onlyfilename=True)

        self.transform = transforms.Compose(
            [
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            ]
        )

        if task == 'All':
            self.X = X
            self.y = y
            self.mask = mask
        else:
            self.X = X[mask[task] == 1]
            self.y[task] = y[task][mask[task] == 1]

        print(len(self.X))

    def __getitem__(self, i):
        img = pil_loader(os.path.join(self.img_path, self.X[i]))
        img = self.transform(img)
        if self.task == 'All':
            return img, self.y['VA'][i], self.y['EX'][i], self.y['AU'][i], self.mask['VA'][i], self.mask['EX'][i], self.mask['AU'][i]
        else:
            return img, self.y[self.task][i]

    def __len__(self):
        return len(self.X)

    def ex_weight(self):
        if self.task == 'EX':
            unique, counts = np.unique(self.y['EX'].astype(int), return_counts=True)
            emo_cw = 1 / counts
            emo_cw/= emo_cw.min()
            return emo_cw
        else:
            return np.zeros(8)

    def au_weight(self):
        if self.task == 'AU':
            weight = 1.0 / np.sum(self.y['AU'], axis=0)
            weight = weight / weight.sum() * self.y['AU'].shape[1]
            return weight
        else:
            return np.zeros(12)