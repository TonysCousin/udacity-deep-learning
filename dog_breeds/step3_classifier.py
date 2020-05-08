########## This is a copy of the code from my Jupyter notebook, taken 5/7/20 from Github branch M3, commit 336f7aa1.
########## It is the code relevant to step 3, build-your-own dog classifier.  Other parts of the notebook, mostly
########## above step 3, are omitted.

########## Cell 1

import numpy as np
import torch

# returns the number of GiB of cuda memory used
def memory_gb():
    gb_alloc = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
    gb_res   = torch.cuda.memory_reserved() /  1024 / 1024 / 1024
    max_val = gb_alloc
    if gb_res > gb_alloc:
        max_val = gb_res
    return max_val

# printable report of cuda memory used
def memory_rpt():
    print("Allocated memory: {:.3f} GB".format(memory_gb()))
    
# finds the index of the element with the largest value in this tensor of shape (1, n)
def max_index(t):
    largest = -1
    largest_index = -1
    for i in range(t.shape[1]):
        if t[0][i] > largest:
            largest = t[0][i]
            largest_index = i
            
    return largest_index
    
memory_rpt()




########## cell 2

import numpy as np
from glob import glob

# load filenames for human and dog images
human_files = np.array(glob("data/lfw/*/*"))
dog_files = np.array(glob("data/dogImages/*/*/*"))

# print number of images in each dataset
print('There are %d total human images.' % len(human_files))
print('There are %d total dog images.' % len(dog_files))
memory_rpt()





########## cell 3 (load the data)

import os
from torchvision import datasets
import torchvision.transforms as transforms

### TODO: Write data loaders for training, validation, and test sets
## Specify appropriate transforms, and batch_sizes
batch_size = 10
train_transform = transforms.Compose([transforms.RandomResizedCrop(224), 
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(20),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])])
val_transform = transforms.Compose([transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder("data/dogImages/train/", transform=train_transform)
val_data = datasets.ImageFolder("data/dogImages/valid/", transform=val_transform)
test_data = datasets.ImageFolder("data/dogImages/test/", transform=val_transform)
print("train_data size = ", len(train_data))
print("val_data size   = ", len(val_data))
print("test_data size  = ", len(test_data))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
test_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

'''
# quick test
# this test confirms that vgg understands the images, so the images are probably correct.
# However, my attempt to get the correct breed classification out of vgg is not working, as
# vgg returns indexes of breeds according to the ImageNet classification (indexes [151, 268])
# but my local data store of just dogs has the classes indexed from [1, 133]. While there are
# names associated with these, it appears there may be some challenges matching the words
# between the two databases.  Not worth my time to solve that problem now, as I have reasonable
# confidence that vgg is interpreting the images correctly, even though I can't seem to display
# them.
total_correct = 0
total_classified = 0
total_samples = 0
for data, labels in val_loader:
    batch_correct = 0
    batch_classified = 0
    #print("   labels = ", labels)
    #print("   label.shape = ", labels.shape)
    for image, label in zip(data, labels):
        im = image.view(1, 3, 224, 224)
        #print("im.shape = ", im.shape)
        output = VGG16(im)
        sm = torch.nn.Softmax(dim=1)
        output = sm(output)
        res = max_index(output)
        # print("    res = ", res, ", label = ", label.item())
        if (res >= 151  and  res <= 268):
            batch_correct += 1
            if (res == label.item()):
                batch_classified += 1
    print("Batch score = {:.1f}%, correctly classified {:.1f}%".format(
          100.0*batch_correct/len(data), 100.0*batch_classified/len(data)))
    total_correct += batch_correct
    total_classified += batch_classified
    total_samples += len(data)
print("Total score = {:.1f}% of {}".format(100.0*total_correct/total_samples, total_samples))
print("Correct classification = {:.1f}%".format(100.0*total_classified/total_samples))
'''

''' 
### below block is not working correctly, perhaps due to misunderstanding of cv2 image loading
import matplotlib.pyplot as plt                        
%matplotlib inline                               

for data, label in train_loader:
    print("data size = ", len(data), ", label size = ", len(label))
    for i in range(len(data)):
        print("\nlabel = ", label[i])
        item = data[i].view(224, 224, -1)
        #item = data[i]
        print("item type = ", type(item))
        print("item shape = ", item.shape)
        item = item.numpy()
        print("numpy item shape = ", item.shape)
        
        ### this is showing the image in each color channel separately, so image is 9 small images
        rgb = cv2.cvtColor(item, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb)
        plt.show()
    break
'''




########## cell 4 (define the model)

import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net(nn.Module):
    ### TODO: choose an architecture, and complete the class
    
    image_size = 224 #both height & width
    image_planes = 3
    
    def __init__(self):
        super(Net, self).__init__()
        ## Define layers of a CNN
        
        # --- making this model overly simple to test cuda garbage collection problems
        
        self.pool = nn.MaxPool2d(2, 2)
        self.c1 = nn.Conv2d(3, 8, 5, padding=2)
        self.c2 = nn.Conv2d(8, 16, 3, padding=1)
        #self.c3 = nn.Conv2d(16, 32, 3, padding=1)
        #self.c4 = nn.Conv2d(32, 48, 3, padding=1)
        #self.c5 = nn.Conv2d(48, 48, 3, padding=1)
        
        #FC layers
        self.dropout = nn.Dropout(0.2)
        #self.fc1 = nn.Linear(37632, 8192)
        #self.fc2 = nn.Linear(8192, 512)
        #self.fc3 = nn.Linear(512, 133)
        
        self.fc3 = nn.Linear(50176, 133)
    
    def forward(self, x):
        num_batches = x.shape[0]
        #print("Entering forward: x.shape = ", x.shape, ", num_batches = ", num_batches)
        
       
        # --- simplified for garbage collection test
        x = F.relu(self.c1(x))        # output 8 x 224 x 224
        x = self.pool(x)              # output 8 x 112 x 112
        x = F.relu(self.c2(x))       # output 16 x 112 x 112
        x = self.pool(x)              # output 16 x 56 x 56
        x = x.view(num_batches, -1) #should be [n, 50176]

        
        ''' skipping my real model below for garbage collection test
        ## Define forward behavior
        x = F.relu(self.c1(x))        # output 8 x 224 x 224
        x = F.relu(self.c2(x))        # output 16 x 224 x 224
        x = self.pool(x)              # output 16 x 112 x 112
        x = F.relu(self.c3(x))        # output 32 x 112 x 112
        x = F.relu(self.c4(x))        # output 48 x 112 x 112
        x = self.pool(x)              # output 48 x 56 x 56
        
        x = F.relu(self.c5(x))        # output 48 x 56 x 56
        x = self.pool(x)              # output 48 x 28 x 28
        
        # flatten the image into a vector, one for each item in the batch
        x = x.view(num_batches, 1, -1)
        #print("Flattened x shape = ", x.shape)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        '''
        
        x = self.fc3(x) #optimizer will apply the Softmax function
        
        return x

#-#-# You do NOT have to modify the code below this line. #-#-#

# instantiate the CNN
model_scratch = Net()

# move tensors to GPU if CUDA is available
use_cuda = torch.cuda.is_available()
if use_cuda:
    model_scratch.cuda()
    print("Model moved to cuda.")
    memory_rpt()




########## cell 5 (check memory usage)

memory_rpt()
torch.cuda.empty_cache()
memory_rpt()
torch.cuda.device_count()
print(torch.cuda.get_device_name())
print(torch.cuda.get_device_capability())
torch.cuda.ipc_collect()
memory_rpt()




########## cell 6 (define loss function & optimizer)

import torch.optim as optim

### TODO: select loss function
criterion_scratch = torch.nn.CrossEntropyLoss() # specifies the Softmax function on the output vector

### TODO: select optimizer
optimizer_scratch = optim.SGD(model_scratch.parameters(), lr=0.001)





########## cell 7 (train & validate)

# the following import is required for training to be robust to truncated images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    training_size = len(loaders['train'].dataset)
    valid_size = len(loaders['valid'].dataset)
    print("Entering training. Num training batches = ", len(loaders['train']),
          ", num validation batches = ", len(loaders['valid']))
    print("  Total training dataset size = ", training_size)
    
    if use_cuda:
        model.cuda()
        print("Moving to CUDA!")
        memory_rpt()
    else:
        print("Using the cpu")
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        
        # target is a 1D vector of [batch_size], representing the index of each dog identified
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            if (batch_idx % 100 == 0):
                print("    Training on batch ", batch_idx, ", memory used = {:.3f} GB".format(memory_gb()))
            #print("target.shape = ", target.shape, ", content =\n", target)
          
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
          
            ## find the loss and update the model parameters accordingly
            
            # --- garbage collection test:  if I comment out 3 lines below
            #          optimizer.zero_grad()
            #          loss.backward()
            #          optimizer.step()
            #     then this loop is identical to the validation loop below, and it blows up.
            #     By various permutations of uncommenting those lines, I determine that it is
            #     the call to loss.backward() that is cleaning up the cuda gargabe in this loop
            #     and preventing memory usage from growing uncontrolled.
            
            optimizer.zero_grad()
            output = model(data) # this is a vector of all breeds showing the probabilities
            #output = output.view(data.size(0), -1) #remove extraneous dimension with size 1
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            train_loss += loss.item()*data.size(0) #mult by batch size since the CrossEntropy calc divides it out
            
            # clean out stale GPU memory
            del data
            del target
            del output
            del loss
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            
        train_loss /= training_size
            
        ######################    
        # validate the model #
        ######################
        model.eval()          #////////////// uncomment this!
        #print(" Beginning validation.")
        for batch_idx, (datav, targetv) in enumerate(loaders['valid']):
            if (batch_idx % 20 == 0):
                print("    Validation on batch ", batch_idx, ", memory used = {:.3f} GB".format(memory_gb()))
            # move to GPU
            if use_cuda:
                datav, targetv = datav.cuda(), targetv.cuda()
            ## update the average validation loss
            outputv = model(datav)
            #outputv = outputv.view(datav.size(0), -1)
            lossv = criterion(outputv, targetv)
            valid_loss += lossv.item()*datav.size(0)
            
            # clean out stale GPU memory
            del datav
            del targetv
            del outputv
            del lossv
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            
        valid_loss /= valid_size
            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        ## TODO: save the model if validation loss has decreased
        if valid_loss < valid_loss_min:
            torch.save(model.state_dict(), 'model_scratch.pt')
            valid_loss_min = valid_loss
            
    # return trained model
    print("///// Training complete.  Memory in use = {:.3f} GB".format(memory_gb()))
    return model


# train the model
num_epochs = 200
loaders_scratch = {"train": train_loader,
                   "valid": val_loader,
                   "test":  test_loader}

model_scratch = train(num_epochs, loaders_scratch, model_scratch, optimizer_scratch, 
                      criterion_scratch, use_cuda, 'model_scratch.pt')

# load the model that got the best validation accuracy
model_scratch.load_state_dict(torch.load('model_scratch.pt'))





