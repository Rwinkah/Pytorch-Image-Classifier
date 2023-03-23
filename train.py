import argparse
def main():
    import torch

    from torch import nn, optim
    from torchvision import models
    from torch.utils.data import DataLoader
    from model_setup import model_setup    
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='Path to the training data')
    parser.add_argument('--save_dir', type=str, help='path to save checkpoints')
    parser.add_argument('--arch', type=str,default='resnet',help='model architecture\nAvailable options are:\n"resnet", "vgg13", "alexnet", "densenet". Default model is resnet because was faster for testing')
    parser.add_argument('--learning_rate', type=float, help='custom learning rate')
    parser.add_argument('--hidden_units', type=int, help='specify the hidden unit size')
    parser.add_argument('--epochs', type=int, help='specify number of epochs for training')
    parser.add_argument('--gpu', type=bool, help='true to use gpu to train model')
    
    args = parser.parse_args()
    
   

    
    # create and reIN
 
    model = model_setup(args)

    
if __name__ == '__main__':
    main()
 
    