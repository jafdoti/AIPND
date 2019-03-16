'''
Testing commands

Basic usage: 
    python predict.py /path/to/image checkpoint
Return top KK most likely classes: 
    python predict.py input checkpoint --top_k 3
Use a mapping of categories to real names: 
    python predict.py input checkpoint --category_names cat_to_name.json  
Use GPU for inference: 
    python predict.py input  --gpu    
    
    
    python predict.py  flowers/test/20/image_04910.jpg checkpoint_d121.pth --category_names cat_to_name.json
'''

import argparse
import torch
from torchvision import datasets, transforms, models
import json
import random
import os

# Libraries needed for Image Preprocessing code
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

    Basic usage: python predict.py /path/to/image checkpoint
         
    Options:
    Return top KK most likely classes: python predict.py input checkpoint --top_k 3
    Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
    Use GPU for inference: python predict.py input checkpoint --gpu  
    """
    # Creates parse
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, default='flowers/test/20/image_04910.jpg', help='Directory path of image to be predicted')
    parser.add_argument('checkpoint', type=str, default='checkpoint_vgg16.pth', help='Checkpoint of trained model')
    parser.add_argument('--top_k', type=int, default=5,help='Number of best predictions to be displayed')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',help='Category to names file')
    parser.add_argument('--gpu', type=bool, default=True, help='True sets device to gpu or False to cpu')

    return parser.parse_args()

# Load a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    # Get original model used
    if(filepath.find("vgg16") > 0):
        model = models.vgg16(pretrained=True) 
    else:
        model = models.densenet121(pretrained=True) 

    # Rebuild with classifier
    model.classifier = checkpoint['classifier']
    model.classifier.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image):
    img_width = 256
    img_height = 256
    crop_width = 224
    crop_height = 224

    pil_image = Image.open(image)
    pil_image = pil_image.resize((img_width,img_height))
    

    left = (img_width - crop_width)/2
    top = (img_height - crop_height)/2
    right = (img_width + crop_width)/2
    bottom = (img_height + crop_height)/2

    pil_image = pil_image.crop((left, top, right, bottom))
    
    np_image = np.array(pil_image)
    np_image = np_image/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    np_image = (np_image - mean)/std
    np_image = np.transpose(np_image, (2, 0, 1))

    return np_image

def predict(image_path, model, topk, gpu):

    if (gpu):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
        
    model.to(device)
    model.eval() 
    
    proc_img = process_image(image_path)
    proc_img=torch.FloatTensor(proc_img)
    proc_img = proc_img.to(device)
    proc_img = proc_img.unsqueeze_(0)

    with torch.no_grad():
        output = model.forward(proc_img)

    probs, top_classes = torch.topk(output, topk)
    probs = probs.exp()    
    probs = probs[0].cpu().numpy() 
    top_classes = top_classes.cpu().numpy()

    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[x] for x in top_classes[0]]

    return probs, top_classes

def main():
    
    in_args = get_input_args()
    print(in_args)
           
    model = load_checkpoint(in_args.checkpoint)

    # Get labels for images
    with open(in_args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    test_img = in_args.image
    
    flower_number = in_args.image.split('/')[2]
    flower_name = cat_to_name[flower_number]
    print("Flower being tested and predicted:",flower_name)
    
    probs, classes = predict(test_img, model, in_args.top_k, in_args.gpu)
    print("Probabilities are" , probs)
    print("Classes are", classes)

    class_names = [cat_to_name[i] for i in classes]
    for i in range(len(class_names)):
        print(class_names[i])
        
        
if __name__ == "__main__":
    main()