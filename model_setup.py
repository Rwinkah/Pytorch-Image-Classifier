import torch
from torch import nn
import torchvision.models as models
from dataprocessing import create_data
from torch import optim
from torch.nn import NLLLoss
from time import time
from torch import cuda
from torchvision import transforms


def model_setup(args):

    """ selecting model based on input
        options are resnet, vgg13, densenet and 
        alexnet because those were the models
        used during the nanodegree 
    """
    

    output_size = 102
    if args.arch == 'resnet':
        model = models.resnet18(pretrained=True)
        input_size = 512
        hid1 = 256
        hid2 = 128

    elif args.arch == 'densenet':
        model = models.densenet121(pretrained=True)
        input_size = 1024
        hid1 = 512
        hid2 = 256

    elif args.arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        input_size = 9216
        hid1 = 4086
        hid2 = 512
         
    else: 
        input_size = 25088
        hid1 = 6272
        hid2 = 1568
        model = models.vgg13(pretrained=True)
    
    if args.hidden_units is not None:
        hid1 = args.hidden_units
    
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(nn.Linear(input_size, hid1),
                               nn.ReLU(),
                               nn.Dropout(p=0.2),
                               nn.Linear(hid1, hid2),
                               nn.ReLU(),
                               nn.Dropout(p=0.2),
                               nn.Linear(hid2, output_size),
                               nn.LogSoftmax(dim=1))
    
    
    if args.arch == 'resnet':
        model.fc = classifier
    else:
        model.classifier = classifier
    layer_list = (input_size, hid1, hid2, output_size)
    return(train_model(model, args, layer_list))

    
def train_model(model, args, layer_list):
    """
    Function to train already initialized machine learning model
    using transfer learning
    Args:
        model: model to be trained
        args:  data passed to the program through the command line
    Return:
        model
    """
    
    # taking arguments from argparse for the model
    learn_rate = args.learn_rate if args.learning_rate else 0.003
    device = 'cuda:0' if cuda.is_available() and args.gpu else 'cpu'
    epochs = args.epochs if args.epochs else 10
    path = args.path
    save_dir = args.save_dir if args.save_dir else '.'
    
    # specifying device to use for training
    model.to(device)
    # defining the optimizer and loss function
    criterion = NLLLoss()
    if args.arch == 'resnet':
        optimizer = optim.Adam(model.fc.parameters(), lr=learn_rate)
    else:
        optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
    
    
    # create generator objects for training and evaluation
    data_dict = create_data(path, 'Train')
    trainloader, validloader = data_dict
    print(trainloader)
    # training the model
    val_loss = 0
    train_loss = 0
    accuracy = 0
    step = 0
    print(f'-------Model----------\nModel arch: {args.arch}\nNumber of epochs: {epochs}\nOptimizer: {optimizer}\nLoss Function: {criterion}')
    print('------------------------------')
    for e in range(epochs):
        # get data from training set
        start = time()
        for images, labels in trainloader:
            # forward propagation
            step += 1
            print(f'---------Training----------\n {step}')
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            train_loss += loss.item()
            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
#              calculate accuracy and validation loss after every 5 steps                  
            if step % 5 == 0:
                with torch.no_grad():
                    model.eval()
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)
                        output = model(images)
                        loss = criterion(output, labels)
                        val_loss += loss.item()
                        prob = torch.exp(output)
                        
                        top_val, top_idx = prob.topk(1, dim=1)
                        equals = top_idx == labels.view(*top_idx.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))
                print(f'epoch : {e +1}')
                print(f'# validation_loss : {val_loss/ len(validloader)}')
                print(f'# accuracy : {accuracy / len(validloader)}')
                print(f'training loss : {train_loss / len(step)}')
        step = 0
        train_loss = 0
        model.train()
        timerr = start - time()
        print(f'time for epoch {e + 1} : {timerr}')
    
    # save the model
    filename = save_dir + '/checkpoint.pth'
    save_model(model, args, filename, layer_list)
    print('Training complete!')
    print('------------------------------')
    print(f'Model saved at {filename}')
    return model
    
                                    
def save_model(model, args, filename, layer_list):                        
    """
    Function to save model_data to a file
    Args:
        model: model to save
        path: save directory
        filename: name of the .pth file
    Return:
        None
    """
 
    last = 'fc' if args.arch == 'resnet' else 'classifier'
                                
    model_data = {'state_dict': model.fc.state_dict(), 'layer_list': layer_list, 'arch':args.arch}
    torch.save(model_data, filename)
    
                                   
        
def load_model(checkpoint):
        
    model_data = torch.load(checkpoint)
    state_dict = model_data['state_dict']
    m_name = model_data['arch']
    print(m_name)
    
    if m_name == 'resnet':
        arch = 'resnet18'
    elif m_name == 'alexnet':
        arch = 'alexnet'
    elif m_name == 'densenet':
        arch = 'densenet121'
    else:
        arch = 'vgg13'
    
    
    
    last = 'fc' if m_name == 'resnet' else 'classifier'
    layer = model_data['layer_list']
    model = models.__dict__[arch](pretrained=True)

    classifier = nn.Sequential(nn.Linear(layer[0], layer[1]),
                               nn.ReLU(),
                               nn.Dropout(p=0.2),
                               nn.Linear(layer[1], layer[2]),
                               nn.ReLU(),
                               nn.Dropout(p=0.2),
                               nn.Linear(layer[2], layer[3]),
                               nn.LogSoftmax(dim=1))
    classifier.load_state_dict(state_dict)
    if m_name == 'resnet':
        model.fc = classifier
    else:
        model.classifier = classifier
    
                               
    return model
     
                               
                                         
                                         