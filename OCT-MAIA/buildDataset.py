import numpy as np
import os
import glob
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import dataset
import pickle
from sklearn.utils import resample
from skimage import exposure
import cv2
import math

def getN(M, L):
    return math.floor((L-1)//(M-1))

def test(M, L):
    idx = list(range(L))
    return len(idx[0::getN(M, L)])

def subsample(ls,M):
    '''Subsampling a list, taking equally distanced samples'''
    L = len(ls)
    N = getN(M, L)
    return ls[0::N][0:M]

def downsample(X,y):
    n = y.shape[0]
    n_positive = (y == 1).sum()
    if (n_positive < n/2):
        mayorityclass = (y == 0)
        minorityclass = (y ==1)
    elif(n_positive > n/2):
        minorityclass = (y == 0)
        mayorityclass = (y ==1)
    else:
        return X,y
    
    #set the minority class to a seperate array
    minority = X[minorityclass]
    #set other classes to another array
    mayority = X[mayorityclass] 
    
    mayor_downsample = resample(mayority,random_state=42,n_samples=minority.shape[0],replace=True)

    X = np.vstack((mayor_downsample,minority))
    y = np.hstack((1-1*y[minorityclass], y[minorityclass]))
    return X,y

def upsample(X,y):
    n = y.shape[0]
    n_positive = (y == 1).sum()
    if (n_positive < n/2):
        mayorityclass = (y == 0)
        minorityclass = (y ==1)
    elif(n_positive > n/2):
        minorityclass = (y == 0)
        mayorityclass = (y ==1)
    else:
        return X,y

    #set the minority class to a seperate array
    minority = X[minorityclass]
    #set other classes to another array
    mayority = X[mayorityclass] 
    #upsample the minority class
    minor_upsampled = resample(minority,random_state=42,n_samples=mayority.shape[0],replace=True)
    #concatenate the upsampled dataframe
    X = np.vstack((minor_upsampled,mayority))
    y = np.hstack((1-1*y[mayorityclass], y[mayorityclass]))
    return X,y

def upsampleWithGroup(X,y,g):
    print(f'initial shapes: X {X.shape}, y {y.shape},g {g.shape}')
    X2 = np.hstack((X,y.reshape((-1, 1))))
    print(f'shape after stack: X2 {X2.shape}')
    X3, _ = upsample(X2,g)
    print(f'shape after first upsample: X3 {X3.shape}, splitted into {X3[:,:-1].shape} and {X3[:,-1].shape}')
    X,y = upsample(X3[:,:-1],X3[:,-1])
    print(f'final shapes: x: {X.shape}, y : {y.shape}')
    return X,y

def downsampleWithGroup(X,y,g):
    X2 = np.hstack((X,y.reshape((-1, 1))))
    X3, _ = downsample(X2,g)
    X,y = downsample(X3[:,:-1],X3[:,-1])
    return X,y
    
def train_test_path_split(root_path,p = 0.8):
    
    def fix(ls):
        ls2 = []
        for elem in (ls):
            if '+' in elem:
                ls2.append(elem.split("+")[0])
                ls2.append(elem.split("+")[1])
            else:
                ls2.append(elem)
        return ls2

    def searchSuboptimalSplit(lsnum,lspaths,target,group):
        '''
        Create two lists of numbers S1 and S2 to suboptimally approach target with the sum of S1 from a list of integers.

            Parameters:
                    lsnum (list(int)): list of integers to split in two lists
                    lspaths (list(str)): list of string to split accordingly to lsnum
                    target: int
                    group: used only for printing

            Returns:
                    trainpaths,testpaths: two sublists of lspaths, consisting of S1 and S2
        '''

        lspaths = [x for _,x in sorted(zip(lsnum,lspaths))]
        lsnum = sorted(lsnum)
        idx = len(lsnum) - 1
        idxs = []
        tot = 0
        while(idx >= 0 and tot < target):
            if(tot + lsnum[idx]  <= target):
                tot += lsnum[idx]
                idxs.append(idx)
            idx -=1
        print(f'the percentage of {group} data, tha ended in the train set is {100*p*tot/target}')
        trainpaths = [lspaths[index] for index in idxs]
        testpaths = [i for i in lspaths if i not in trainpaths]
        return fix(trainpaths),fix(testpaths)
            
        
    paths =  glob.glob(os.path.join(root_path + '\*.pickle'))
    paths_amd = []
    paths_control = []
    num_amd = []
    num_control = []
    total_amd = 0
    total_control = 0
    iterator = iter(range(0,len(paths)))
    
    for i in iterator:
        # Check if the two next paths are of the same patient (left and right eye), if the case we treat them as one
        two_eyes_same_patient = paths[i].split("\\")[-1].split("_")[:-1]==paths[i+1].split("\\")[-1].split("_")[:-1]
        if(two_eyes_same_patient):
            PPocts = pd.read_pickle(paths[i]) + pd.read_pickle(paths[i + 1])
            path = paths[i] + '+' + paths[i+1]
            next(iterator, None)

        else:
            PPocts = pd.read_pickle(paths[i])
            path = paths[i]
            
        group = path.split("\\").pop().split("_")[0]
        num_octs = len(PPocts)
        
        if (group == 'amd'):
            paths_amd.append(path)
            num_amd.append(num_octs)
            total_amd += num_octs
            
        elif (group == 'control'):
            paths_control.append(path)
            num_control.append(num_octs)
            total_control += num_octs
    
    #Desired amd OCT's in the train
    train_amd = int(p*total_amd)   
    train_amd,test_amd = searchSuboptimalSplit(num_amd,paths_amd,train_amd,group = 'amd')
    
    #Desired control OCT's in the train
    train_control = int(p*total_control)
    train_control,test_control = searchSuboptimalSplit(num_control,paths_control,train_control,group = 'control')
    
    return train_amd+train_control,test_amd+test_control

#HOW TO CALL THIS FUNCTION
# root_path =      r'C:\Users\line\Desktop\Mauro\preprocessing\preprocessed_octs' 
# train,test = train_test_path_split(root_path)  

DATAPATHS = {
    "preprocessed": r'C:\Users\line\Desktop\Mauro\preprocessing\preprocessed_octs',
    "rootsavepath": r'C:\Users\line\Desktop\Mauro\3_DataSet\Vectors' ,
}
def getXYdata(paths, mode,rootpath = None,normmode = 'Z-score'):
    def normalize(img,normmode = 'Z-score'):
        if normmode == 'Z-score':
            img = dataset.z_score(img)
        elif normmode == 'EQ-hist': 
            img = exposure.equalize_hist(img)
        elif normmode == 'CLAHE':
            #TO Implement normalization by zone
            # Adaptive Equalization
            img = exposure.equalize_adapthist(img/max((img.max(),-img.min())), clip_limit=0.03)            
        return img
    
    def getxmin(rootpath):
        themin = 100000
        allpaths =  paths
        for path in paths:
            PPocts = pd.read_pickle(path)
            for PPoct in PPocts:
                xnew = PPoct.image()
                if(xnew.shape[0] < themin):
                    themin = xnew.shape[0]
        return themin
    
    if mode == 'thickness':
        xmin = 5 
    elif mode =='raw':
        xmin = getxmin(rootpath)
        
    X = np.empty((0,xmin))
    y = np.empty((0,))
    for path in paths:
        PPocts = pd.read_pickle(path)
        for PPoct in PPocts:
            if mode == 'thickness':
                xnew = np.transpose(PPoct.thicknesses())
            elif mode == 'raw':
                xnew = np.transpose(normalize(PPoct.image(),normmode = normmode))
                xnew = cv2.resize(xnew, dsize=(xmin,768 ))
            ynew = PPoct.label()
            #stack vertically the x and y
            X = np.vstack((X,xnew))
            y = np.hstack((y,ynew))          
      
    return X,y
    
def getBalancedXYData(mode = 'thickness',normmode = 'EQ-hist'):
    rootpath = r'C:\Users\line\Desktop\Mauro\3_DataSet\OCT_balanced'
    paths = glob.glob(os.path.join(rootpath,'controlP' + '\*.pickle'))
    paths2 = glob.glob(os.path.join(rootpath,'amdP' + '\*.pickle'))
    merged = [[paths[2*i],paths[2*i+1],paths2[i]]  for i in range(len(paths2))]
    flattenedmerged = [item for sublist in merged for item in sublist]
    paths = flattenedmerged
    return getXYdata(paths, mode = mode,rootpath = rootpath,normmode = normmode)

def create_dataset(root_path,mode ='thickness',normmode = 'Z-score'):
    trainpaths,testpaths = train_test_path_split(root_path) 
    #print([path.split('\\')[-1] for path in trainpaths],'\n',[path.split('\\')[-1] for path in testpaths])
    # if any files in the folder delete them
    X_train,y_train,group_train = getXYdata(trainpaths,mode,root_path,normmode)
    X_test,y_test,group_test = getXYdata(testpaths,mode,root_path,normmode)
    return X_train, X_test, y_train, y_test,group_train,group_test

def save_dataset(root_path,mode ='thickness',normmode = 'Z-score', samplingmode = '',onlyCtrl = False,onlyAMD = False):
    def getpath(mode,normmode,samplingmode,onlyCtr = False,onlyAMD = False,):
        if(onlyCtr or onlyAMD):
            return f'{DATAPATHS["rootsavepath"]}\\{mode}{normmode}{samplingmode}{onlyCtr}{onlyAMD}.pkl'
        return f'{DATAPATHS["rootsavepath"]}\\{mode}{normmode}{samplingmode}.pkl'
    save_path = getpath(mode,normmode,samplingmode,onlyCtrl,onlyAMD)
    X_train, X_test, y_train, y_test,group_train,group_test = create_dataset(root_path,mode,normmode)
    #Groupvect is one if amd
    if(onlyCtrl):
        X_train = X_train[~group_train.astype(bool)]
        X_test = X_test[~group_test.astype(bool)]
        y_train = y_train[~group_train.astype(bool)]
        y_test = y_test[~group_test.astype(bool)]
    elif(onlyAMD):
        X_train = X_train[group_train.astype(bool)]
        X_test = X_test[group_test.astype(bool)]
        y_train = y_train[group_train.astype(bool)]
        y_test = y_test[group_test.astype(bool)]
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    if samplingmode == 'U':
        print('UPSAMPLING')
        X_train,y_train = upsample(X_train,y_train)#upsampleWithGroup(X_train,y_train,group_train)
    elif samplingmode == 'D':
        print('DOWNSAMPLING')
        X_train,y_train  = downsample(X_train,y_train)#downsampleWithGroup(X_train,y_train,group_train)
                                      
    with open(save_path,'wb') as f:
        pickle.dump(X_train,f)
        pickle.dump(X_test,f)
        pickle.dump(y_train,f)
        pickle.dump(y_test,f)

def open_dataset(mode ='thickness',normmode = 'Z-score', samplingmode = '',onlyCtrl = False,onlyAMD = False):
    if(mode == 'thickness' and normmode != ''):
        print('WARNING: normalizing the thicknesses is not a good idea since they represent physical values')
    def getpath(mode,normmode,samplingmode,onlyCtr = False,onlyAMD = False,):
        if(onlyCtr or onlyAMD):
            return f'{DATAPATHS["rootsavepath"]}\\{mode}{normmode}{samplingmode}{onlyCtr}{onlyAMD}.pkl'
        return f'{DATAPATHS["rootsavepath"]}\\{mode}{normmode}{samplingmode}.pkl'
    path = getpath(mode,normmode,samplingmode,onlyCtrl,onlyAMD)
    with open(path,'rb') as f:
        X_train = pickle.load(f)
        X_test = pickle.load(f)
        y_train = pickle.load(f)
        y_test = pickle.load(f)
    return X_train, X_test, y_train, y_test

def saveFullImgDataset(paths,lblOutpath,ImgsOutpath,rootpath = None,normmode = 'Z-score'):
    def normalize(img,normmode = 'Z-score'):
        if normmode == 'Z-score':
            img = dataset.z_score(img)
        elif normmode == 'EQ-hist': 
            img = exposure.equalize_hist(img)
        elif normmode == 'CLAHE':
            #TO Implement normalization by zone
            # Adaptive Equalization
            img = exposure.equalize_adapthist(img/max((img.max(),-img.min())), clip_limit=0.03)            
        return img
    
    def getxmin(rootpath):
        themin = 100000
        allpaths =  glob.glob(os.path.join(rootpath,'amdP' + '\*.pickle')) + glob.glob(os.path.join(rootpath,'controlP' + '\*.pickle'))
        for path in allpaths:
            PPocts = pd.read_pickle(path)
            for PPoct in PPocts:
                xnew = PPoct.image()
                if(xnew.shape[0] < themin):
                    themin = xnew.shape[0]
        return themin
    

    xmin = getxmin(rootpath) 
    y = np.empty((0,))
    
    cnt = 0
    #save image in the output folder, save label in the label folder
    for path in paths:
        PPocts = pd.read_pickle(path)
        for PPoct in PPocts:
            print(cnt)
            #save full img to imgoutpath location
            img = normalize(PPoct.image(),normmode = normmode)
            img = cv2.resize(img, dsize=(768, xmin))
            imgpath = os.path.join(ImgsOutpath,str(cnt))
            cv2.imwrite(f'{imgpath}.png',img) 
            # label into np array
            ynew = PPoct.label()      
            y = np.hstack((y,ynew))
            print(y.shape,img.shape)
            cnt += 1
    # Save labels np array to the labelspath
    with open(lblOutpath,'wb') as f:
        pickle.dump(y,f)
        
def keep_good_segmentation(PPocts,special = False):
        "Filter data to avoid the issue of wrong segmentation"
        filtered = []
        for PPoct in PPocts:
            if PPoct.image().shape[0] < 120:
                filtered.append(PPoct)
            elif special:
                filtered.append(PPoct)      
        return filtered

def save_balancedPatients_octs(root_path):     
    paths =  glob.glob(os.path.join(root_path + '\*.pickle'))
    paths_amd = []
    paths_control = []
    num_amd = []
    num_control = []
    rateoAmd = []
    rateoControl = []
    total_amd = 0
    total_control = 0
    iterator = iter(range(0,len(paths)))
    AMDtwoEyesSamePatient = []
    CtrltwoEyesSamePatient = []

    
    for i in iterator:
        path = paths[i]
        # amd or control
        group = path.split("\\").pop().split("_")[0]
        # what's the id?
        Id = path.split("\\").pop().split("_")[1]
        special  = Id == '12' and group != 'amd'
        print(group,Id == '12', group != 'amd')
        if special:
            print(len(pd.read_pickle(path)))
        # left or right?
        side = path.split("\\").pop().split("_")[2]
        PPocts = pd.read_pickle(path)
        PPocts = keep_good_segmentation(PPocts,special)
        #SubSample randomly according to dicts
        if group == 'amd':  
            N = dictAMD[path]
            
        else:
            N = dictCtrl[path]
            
        outPath = os.path.join(r'C:\Users\line\Desktop\Mauro\3_DataSet\OCT_balanced',group,Id + '_' + side) 
        print(outPath)
        print(f'eye: {group}{Id}{side} from {len(PPocts)} to {len(subsample(PPocts,N))}')
        # IF you want to sample randomly
        #PPocts = random.sample(PPocts,N)
        # If you want to maximize variety

        PPocts = subsample(PPocts,N)
        
        with open(outPath, 'wb') as f:
            pickle.dump(PPocts, f)

