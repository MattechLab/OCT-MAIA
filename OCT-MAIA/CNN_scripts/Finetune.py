#imports
import tqdm
import os
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib
from torchvision import models as models
import torch.nn as nn
# from engine import train, validate
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import KFold
matplotlib.style.use('ggplot')
import numpy as np
import pickle
import cv2

# initialize the computation device and paths
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class ImageDataset(Dataset):
    def __init__(self,octpath, lblpath,width,stride,transform):
        self.octpath = octpath
        self.width = width
        self.stride = stride
        self.transform = transform
        with open(lblpath,'rb') as f:
            self.labels = (np.reshape(pickle.load(f),(-1,768))[:,width//2:768 - width//2:stride]).flatten()

    def __len__(self):
        return len(self.labels)
    
    def get_indexes(self,index):
        N_oct_image = (768 - self.width)//self.stride + 1
        path_N = index // N_oct_image
        insideIdx = index % N_oct_image
        return path_N,insideIdx,N_oct_image
    
    def get_data(self, index):
        imgidx,insideIdx,N_oct_image = self.get_indexes(index)
        path = os.path.join(self.octpath,str(imgidx))
        image = cv2.imread(f"{path}.png")
        image = image[:,self.stride*insideIdx : self.stride*insideIdx+self.width]
        target = self.labels[imgidx*N_oct_image + insideIdx]
        return image,target
    
    def __getitem__(self, index):
        # IMAGE,LABEL 
        image,target = self.get_data(index)
        # PAD WITH BLACK STRIPES ON TOP AND ON THE BOTTOM
        newimage = np.zeros((image.shape[0]*3,image.shape[1],image.shape[2]),dtype = 'uint8')
        newimage[image.shape[0]:image.shape[0]*2,:,:] =  image
        # apply image transforms
        newimage = self.transform(newimage)        
        return (newimage.type(torch.float32),torch.tensor(target, dtype=torch.long) )
    
# Function definitions
def save_model(epochs, model, optimizer, criterion,foldername,fold):
    """
    Function to save the trained model to disk.
    """
    path = '/data/line/mauro/outputs/' + foldername + '/' + 'fold' + str(fold) + '/'

    try:
        os.makedirs(path, exist_ok = True)
        print("Directory '%s' created successfully" % path)
    except OSError as error:
        print("Directory '%s' can not be created" % path)
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, path + '/model.pth')
    return path
def save_plots(train_acc, valid_acc, train_loss, valid_loss,path):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(path + 'accuracy.png')
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path + 'loss.png')

def modelBuilder(pretrained, requires_grad,N_output = 2,fc = None):
    """
    Function to build a model with resnet50 architecture.
    pretrained: If true is pretrained on Imagenet dataset
    N_output: Number of output of classification head
    fc: classification head (fully connected)
    """
    model = models.resnet50(progress=True, pretrained=pretrained)
    # to freeze the hidden layers
    if requires_grad == False:
        for param in model.parameters():
            param.requires_grad = False
    # to train the hidden layers
    elif requires_grad == True:
        for param in model.parameters():
            param.requires_grad = True
    # make the classification head learnable
    if fc is None:   
        model.fc = nn.Sequential(
                   nn.Linear(2048, 128),
                   nn.ReLU(inplace=True),
                   nn.Linear(128, N_output)).to(device)
    else:
        model.fc = fc
    return model

def load_weights(model,filepath):
    """
    Function to load the pretrained model weights from disk.
    """
    state = torch.load(filepath)
    model.load_state_dict(state['model_state_dict'])
    return model

# training function
def train(model, dataloader, optimizer, criterion, device):
    """
    Model training function.
    """
    print('Training')
    model.train()
    counter = 0
    train_running_loss = 0.0
    train_running_correct = 0
    for i, data in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        counter += 1
        inputs,labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        # apply sigmoid activation to get all the outputs between 0 and 1
        outputs = torch.sigmoid(outputs)
        loss = criterion(outputs, F.one_hot(labels,num_classes= 2).to(torch.float32).to(device))
        train_running_loss += loss.item()
        #accuracy computation
        _, preds = torch.max(outputs, 1)
        train_running_correct += (preds == labels).sum().item()/(labels.size(0))
        # backpropagation
        loss.backward()
        # update optimizer parameters
        optimizer.step()
        
    train_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / counter)
    return train_loss ,epoch_acc

# validation function
def validate(model, dataloader, criterion, device):
    """
    Model validation function.
    """
    print('Validating')
    model.eval()
    counter = 0
    val_running_loss = 0.0
    valid_running_correct = 0
    with torch.no_grad():
        for i, data in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            counter += 1
            inputs,labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            # apply sigmoid activation to get all the outputs between 0 and 1
            outputs = torch.sigmoid(outputs)
            loss = criterion(outputs, F.one_hot(labels,num_classes= 2).to(torch.float32).to(device))
            val_running_loss += loss.item()
            # calculate the accuracy
            _, preds = torch.max(outputs, 1)
            valid_running_correct += (preds == labels).sum().item()/(labels.size(0))
        val_loss = val_running_loss / counter
        epoch_acc = 100. * (valid_running_correct / counter)
        return val_loss, epoch_acc
    
def main():
    # DataLoader

    # related transformation defination
    IMAGE_NET_MEANS = [0.485, 0.456, 0.406]
    IMAGE_NET_STDEVS = [0.229, 0.224, 0.225]

    # the training transforms
    train_transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((224,224)),
                        transforms.Lambda(lambda x: x.convert('RGB')),
                        transforms.ToTensor(),
                        transforms.Normalize(IMAGE_NET_MEANS,IMAGE_NET_STDEVS)
                    ])

    # the validation transforms
    valid_transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((224,224)),
                        transforms.Lambda(lambda x: x.convert('RGB')),
                        transforms.ToTensor(),
                        transforms.Normalize(IMAGE_NET_MEANS,IMAGE_NET_STDEVS)
                    ])

    # learning parameters
    N_output = 2
    lr = 0.0001
    epochs = 20
    batch_size = 32
    save_model = False
    
    train_all = False
    width = 224
    stride = 15
    transform = train_transform
    fc = nn.Sequential(
                   nn.Linear(2048, 128),
                   nn.ReLU(inplace=True),
                   nn.Linear(128, N_output)).to(device)
    
    #intialize the model
    model = modelBuilder(pretrained=True, requires_grad=train_all,N_output = N_output,fc = fc).to(device)
    #Fill with weight trained on octs
    weightpath = '/data/line/mauro/outputs/10000lr_1.0batchsize_32trainall_TrueaugmentFalse/model.pth'
    model = load_weights(model,weightpath) 
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    #LOGS
    print(f"Computation device: {device}\n")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    
    splits = KFold(n_splits = 6, shuffle = False)
    octpath = '/data/line/mauro/octs'
    lblpath = '/data/line/mauro/labels/labels.pickle'
    dataset = ImageDataset(octpath, lblpath,width,stride,transform)
    class_weight = torch.FloatTensor([len(dataset)/(dataset.labels==0).sum(), len(dataset)/(dataset.labels==1).sum()]).to(device)
    criterion = nn.BCELoss(weight = class_weight)
    for fold, (train_idx, valid_idx) in enumerate(splits.split(dataset)):
        print(f'Starting computation for fold: {fold}')
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        #data iterator defination
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler) #,drop_last = True
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler) # ,drop_last = True

        # lists to keep track of losses and accuracies
        train_loss, valid_loss = [], []
        train_acc, valid_acc = [], []
        # start the training
        for epoch in range(epochs):
            print(f"[INFO]: Epoch {epoch+1} of {epochs}")
            train_epoch_loss,train_epoch_acc = train(model, train_loader, 
                                                      optimizer, criterion,device)
            valid_epoch_loss,valid_epoch_acc = validate(model, val_loader,  
                                                         criterion,device)
            train_loss.append(train_epoch_loss)
            valid_loss.append(valid_epoch_loss)
            train_acc.append(train_epoch_acc)
            valid_acc.append(valid_epoch_acc)
            print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
            print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
            print('-'*50)    
        foldername = f'FINETUNElr10000_{lr*10000}batchsize_{batch_size}trainall_{train_all}fold_{fold}'
        # save the trained model weights
        if save_model:
            path = save_model(epochs, model, optimizer, criterion,foldername,fold)
        else:
            path = '/data/line/mauro/outputs/' + foldername + '/' + 'fold' + str(fold) + '/'
            try:
                os.makedirs(path, exist_ok = True)
                print("Directory '%s' created successfully" % path)
            except OSError as error:
                print("Directory '%s' can not be created" % path)
        # save the loss and accuracy plots
        save_plots(train_acc, valid_acc, train_loss, valid_loss,path)
        train_loss, valid_loss = [], []
        train_acc, valid_acc = [], []
        
    print('TRAINING COMPLETE')
    
if __name__ == '__main__':
    main()