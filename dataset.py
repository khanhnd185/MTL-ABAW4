import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight
from torchvision import transforms

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def get_dataset(filename, feature_dict, onlyfilename=False):
    with open(filename) as f:
        mtl_lines = f.read().splitlines()
    num_missed=0
    X,y_va,y_expr,y_aus=[],[],[],[]
    masks_va,masks_expr,masks_aus=[],[],[]
    for line in mtl_lines[1:]:
        splitted_line=line.split(',')
        imagename=splitted_line[0]
        valence=float(splitted_line[1])
        arousal=float(splitted_line[2])
        expression=int(splitted_line[3])
        aus=list(map(int,splitted_line[4:]))
        
        mask_VA=(valence>-5 and arousal>-5)
        if not mask_VA:
            valence=arousal=0
            
        mask_expr=(expression>-1)
        if not mask_expr:
            expression=0
            
        mask_aus=min(aus)>=0
        if not mask_aus:
            aus=[0]*len(aus)
        if mask_VA or mask_expr or mask_aus:
            if imagename in feature_dict or onlyfilename:
                #X.append(filename2featuresAll[imagename][0])
                if onlyfilename:
                    X.append(imagename)
                else:
                    X.append(np.concatenate((feature_dict[imagename][0],feature_dict[imagename][1])))
                y_va.append((valence,arousal))
                masks_va.append(mask_VA)
                
                y_expr.append(expression)
                masks_expr.append(mask_expr)
                
                y_aus.append(aus)
                masks_aus.append(mask_aus)
            else:
                num_missed+=1
    X=np.array(X)
    y_va=np.array(y_va)
    y_expr=np.array(y_expr)
    y_aus=np.array(y_aus)
    masks_va=np.array(masks_va).astype(np.float32)
    masks_expr=np.array(masks_expr).astype(np.float32)
    masks_aus=np.array(masks_aus).astype(np.float32)
    print(X.shape,y_va.shape,y_expr.shape,y_aus.shape,masks_va.shape,num_missed)

    return X,y_va,y_expr,y_aus,masks_va,masks_expr,masks_aus

def get_input(filename, feature_dict, onlyfilename=False):
    with open(filename) as f:
        mtl_lines = f.read().splitlines()
    num_missed=0
    X,y_va,y_expr,y_aus=[],[],[],[]
    masks_va,masks_expr,masks_aus=[],[],[]
    for line in mtl_lines[1:]:
        splitted_line=line.split(',')
        imagename=splitted_line[0]
        valence=float(splitted_line[1])
        arousal=float(splitted_line[2])
        expression=int(splitted_line[3])
        aus=list(map(int,splitted_line[4:]))
        
        mask_VA=(valence>-5 and arousal>-5)
        if not mask_VA:
            valence=arousal=0
            
        mask_expr=(expression>-1)
        if not mask_expr:
            expression=0
            
        mask_aus=min(aus)>=0
        if not mask_aus:
            aus=[0]*len(aus)
        if mask_VA or mask_expr or mask_aus:
            if imagename in feature_dict or onlyfilename:
                #X.append(filename2featuresAll[imagename][0])
                if onlyfilename:
                    X.append(imagename)
                else:
                    X.append(np.concatenate((feature_dict[imagename][0],feature_dict[imagename][1])))
                y_va.append((valence,arousal))
                masks_va.append(mask_VA)
                
                y_expr.append(expression)
                masks_expr.append(mask_expr)
                
                y_aus.append(aus)
                masks_aus.append(mask_aus)
            else:
                num_missed+=1
    X=np.array(X)
    y_va=np.array(y_va)
    y_expr=np.array(y_expr)
    y_aus=np.array(y_aus)
    masks_va=np.array(masks_va).astype(np.float32)
    masks_expr=np.array(masks_expr).astype(np.float32)
    masks_aus=np.array(masks_aus).astype(np.float32)
    print(X.shape,y_va.shape,y_expr.shape,y_aus.shape,masks_va.shape,num_missed)

    return X,y_va,y_expr,y_aus,masks_va,masks_expr,masks_aus

class ImageSAW2(Dataset):
    def __init__(self, filename, img_path):
        super(ImageSAW2, self).__init__()
        with open(filename) as f:
            mtl_lines = f.read().splitlines()
        self.X = mtl_lines[1:]
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

# s-Aff-Wild2
class SAW2(Dataset):
    def __init__(self, filename, feature_dict):
        super(SAW2, self).__init__()
        self.X , self.y_va, self.y_ex, self.y_au, self.mask_va, self.mask_ex, self.mask_au = get_dataset(filename, feature_dict)
    
    def __getitem__(self, i):
        return self.X[i] , self.y_va[i], self.y_ex[i], self.y_au[i], self.mask_va[i], self.mask_ex[i], self.mask_au[i]

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