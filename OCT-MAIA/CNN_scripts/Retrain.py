#SCRIPT VERSION
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
matplotlib.style.use('ggplot')
from datasets import load_dataset


# initialize the computation device and paths
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HFDataset(Dataset):
    def __init__(self, dset,transform = None):
        self.dset = dset
        self.transform = transform

    def __getitem__(self, idx):
        img,label = (self.dset[idx]['image'],self.dset[idx]['label'])
        if self.transform:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.dset)

# Function definitions
def save_model(epochs, model, optimizer, criterion,foldername):
    """
    Function to save the trained model to disk.
    """
    path = '/data/line/mauro/outputs/' + foldername + '/'
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

def modelBuilder(pretrained, requires_grad,N_output = 2):
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
    model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, N_output)).to(device)
    return model

# training function
def train(model, dataloader, optimizer, criterion, device):
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
        loss = criterion(outputs, F.one_hot(labels).to(torch.float32).to(device))
        train_running_loss += loss.item()
        #accuracy computation
        _, preds = torch.max(outputs, 1)
        train_running_correct += (preds == labels).sum().item()
        # backpropagation
        loss.backward()
        # update optimizer parameters
        optimizer.step()
        
    train_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(dataloader.dataset))
    return train_loss ,epoch_acc

# validation function
def validate(model, dataloader, criterion, device):
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
            loss = criterion(outputs, F.one_hot(labels).to(torch.float32).to(device))
            val_running_loss += loss.item()
            # calculate the accuracy
            _, preds = torch.max(outputs, 1)
            valid_running_correct += (preds == labels).sum().item()
        val_loss = val_running_loss / counter
        epoch_acc = 100. * (valid_running_correct / len(dataloader.dataset))
        return val_loss, epoch_acc
    
def main():
    # DataLoader

    # related transformation defination
    IMAGE_NET_MEANS = [0.485, 0.456, 0.406]
    IMAGE_NET_STDEVS = [0.229, 0.224, 0.225]

    # the training transforms
    train_transform = transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.Lambda(lambda x: x.convert('RGB')),
                        transforms.ToTensor(),
                        transforms.Normalize(IMAGE_NET_MEANS,IMAGE_NET_STDEVS)
                    ])


    # the validation transforms
    valid_transform = transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.Lambda(lambda x: x.convert('RGB')),
                        transforms.ToTensor(),
                        transforms.Normalize(IMAGE_NET_MEANS,IMAGE_NET_STDEVS)
                    ])

    # learning parameters
    N_output = 2
    lr = 0.005
    epochs = 20
    batch_size = 32
    augmentdata = True
    criterion = nn.BCELoss()
    train_all = True

    if augmentdata:
        train_transform = transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.Lambda(lambda x: x.convert('RGB')),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomVerticalFlip(p=0.5),
                        transforms.RandomRotation(degrees=(30, 70)),
                        transforms.ToTensor(),
                        transforms.Normalize(IMAGE_NET_MEANS,IMAGE_NET_STDEVS)
                    ])

    
    #intialize the model
    model = modelBuilder(pretrained=True, requires_grad=train_all,N_output = N_output).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    #LOGS
    print(f"Computation device: {device}\n")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    dataset = load_dataset("MauroLeidi/OCT_balanced")

    # Data loading and labeling
    train_data = HFDataset(dataset['train'],train_transform)
    val_data = HFDataset(dataset['test'],valid_transform)

    #data iterator defination
    train_loader = DataLoader(train_data,
                                shuffle = True,
                                batch_size=batch_size)

    val_loader = DataLoader(val_data,
                              shuffle = True,
                              batch_size=batch_size)

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
    foldername = f'10000lr_{lr*10000}batchsize_{batch_size}trainall_{train_all}'
    # save the trained model weights
    path = save_model(epochs, model, optimizer, criterion,foldername)
    # save the loss and accuracy plots
    save_plots(train_acc, valid_acc, train_loss, valid_loss,path)
    print('cache cleanup')
    dataset.cleanup_cache_files()
    print('TRAINING COMPLETE')

if __name__ == '__main__':
    main()