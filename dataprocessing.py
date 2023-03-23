def create_data(path, train_mode):
    """
    Create dataloader object for training, validation and testing
    Args:
        path: path to folder containing all image folders
        train_mode: returns dataloaders for training if true
    Return:
        dict
    """
    
    from torchvision import transforms, datasets
    from torch.utils.data import DataLoader
    from PIL import Image
    # make transforms for creating datasets
    train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.CenterCrop(224),
                                          transforms.RandomRotation((1, 30)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([.485, .456, .406],
                                                               [.229, .224, .225])])

    test_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([.485, .456, .406],
                                                             [.229, .224, .225])])

    # create dataset
    train_dataset = datasets.ImageFolder((path + '/train'), train_transform)

    valid_dataset = datasets.ImageFolder((path + '/valid'    ), test_transform)

    test_dataset = datasets.ImageFolder((path + '/test'), test_transform)
    
    # create dataloaders
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validloader = DataLoader(valid_dataset, batch_size=32,)
    
    
    if train_mode == 'Train':
        return trainloader, validloader
    elif train_mode == 'Test':
        return(test_transform)
    
    
    
    
    
    
    