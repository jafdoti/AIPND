'''
Testing commands

Basic usage: 
    python train.py data_directory
Set directory to save checkpoints: 
    python train.py flowers --save_dir 
Choose architecture: 
    python train.py flowers --arch "vgg16"
Set hyperparameters: 
    python train.py flowers --learning_rate 0.001 --hidden_units 512 --epochs 5
Use GPU for training: 
    python train.py flowers --gpu    
    
    python train.py flowers --arch "vgg16" --learning_rate 0.001 --hidden_units 512 --epochs 5
    
    python train.py flowers --arch "densenet" --learning_rate 0.01 --hidden_units 64 --epochs 20
'''


import argparse
import datetime
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json
from PIL import Image
import numpy as np


# Functions defined below
def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object
     
    Options:
    Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
    Choose architecture: python train.py data_dir --arch "vgg13"
    Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
    Use GPU for training: python train.py data_dir --gpu     
    """
    # Creates parse
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='', help='Checkpoint save directory')
    parser.add_argument('data_dir', type=str, default='flowers', help='path to folder of flower images')
    parser.add_argument('--arch', type=str, default='vgg16',help='chosen model')
    parser.add_argument('--learning_rate', type=float, default='.001',help='Model learning rate')
    parser.add_argument('--hidden_units', type=int, default='512',help='Hidden units in classifier')
    parser.add_argument('--epochs', type=int, default='10',help='Number of training epochs')
    parser.add_argument('--gpu', type=bool, default=True,help='Sets device to cpu or gpu for training')

    return parser.parse_args()

def validation(model, validationloader, criterion, device):
    test_loss = 0
    accuracy = 0
    for images, labels in validationloader:

        # Move input and label tensors to the GPU
        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy



def main():
    
    in_args = get_input_args()

    base_save_dir = ""
    if(in_args.save_dir != ''):
        base_save_dir = in_args.save_dir + "/" 
    
    training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),
                                                        (0.229, 0.224, 0.225))])
    
    validation_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    
    testing_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    
    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(in_args.data_dir + '/train', transform=training_transforms)
    validate_data = datasets.ImageFolder(in_args.data_dir + '/valid', transform=validation_transforms)
    test_data = datasets.ImageFolder(in_args.data_dir + '/test', transform=testing_transforms)
    
    # Using the image datasets and the trainforms, define the dataloaders
    training_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validate_loader = torch.utils.data.DataLoader(validate_data, batch_size=64)
    testing_loader = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    checkpoint_path = ""
    input_size = 0    
    if(in_args.arch == "vgg16"):
        model = models.vgg16(pretrained=True)
        input_size = 25088
        checkpoint_path = base_save_dir + "checkpoint_vgg16.pth"
    elif(in_args.arch == "densenet"):
        model = models.densenet121(pretrained=True)
        input_size = 1024
        checkpoint_path = base_save_dir + "checkpoint_d121.pth"
    
    # Hyperparameters for our network classifier
    hidden_size = in_args.hidden_units
    output_size = 102
    dropout_rate = 0.2
    
    # Freeze parameters so we don't backprop through them
    # Loop through all the parameters in the model
    # And halt gradient calculations
    # Will not be trained during training steps
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_size, hidden_size)),
                              ('relu1', nn.ReLU()),
                              ('drop1', nn.Dropout(dropout_rate)),
                              ('fc2', nn.Linear(hidden_size, output_size)),  
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
        
    # Replace DenseNet classifier with this classifier
    model.classifier = classifier
    
    # Set criterion and weights
    # Need to reference custom classifier parameters
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=in_args.learning_rate)
    
    # Device agnostice code
    if (in_args.gpu):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    model.to(device)
    #epochs = 4
    steps = 0
    running_loss = 0
    print_every = 40
    print("Testing", in_args.arch, "on", device)
    for e in range(in_args.epochs):
        print ("EPOCH", e+1, "AT", datetime.datetime.now().strftime("%H:%M:%S"))
        model.train()
        for images, labels in training_loader:
            
            steps += 1
                        # Move input and label tensors to the GPU
            images, labels = images.to(device), labels.to(device)

            # Zero out weights
            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    validation_loss, accuracy = validation(model, validate_loader, criterion, device)

                print("Epoch: {}/{}.. ".format(e+1, in_args.epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(validation_loss/len(validate_loader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validate_loader)))

                running_loss = 0
                model.train()
    print("Training finished at ", datetime.datetime.now().strftime("%H:%M:%S"))
    
    with torch.no_grad():
        test_loss, accuracy = validation(model, testing_loader, criterion, device)
    
    print("Test Loss: {:.3f}.. ".format(test_loss/len(testing_loader)),
          "Test Accuracy: {:.3f}".format(accuracy/len(testing_loader)))

    # Save the checkpoint 
    checkpoint = {'input_size': input_size,
                  'output_size': output_size,
                  'hidden_layers': hidden_size,
                  'state_dict': model.classifier.state_dict(),
                  'dropout': dropout_rate,
                  'classifier': model.classifier,
                  'class_to_idx': train_data.class_to_idx,
                  'optimizer state': optimizer.state_dict
                 }
    
    torch.save(checkpoint, checkpoint_path)    
    
if __name__ == "__main__":
    main()