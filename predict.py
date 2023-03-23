def main():
    """
    Entry point, using a saved model for prediction
    Args:
        path: path to image
        checkpoint: saved model .pth file
        --top_k: specifies how many classes to output from prediction
        --gpu: specifies whether inference is done on gpu or cpu
        --category_names: json file for class names
    Rerurn:
        class names and probabilities
    """
    
    import argparse
    import model_setup
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="path to the image to be classified")
    parser.add_argument('checkpoint', type=str, help="path to trained model")
    parser.add_argument('--top_k', default=3, type=int, help="specify amount of classifications")
    parser.add_argument('--gpu', type=bool, default=True, help="specify true to run inference on gpu")
    parser.add_argument('--category_names', default='cat_to_name.json', type=argparse.FileType("r"), help="path to JSON")
    
    
    args = parser.parse_args()
    
    
    
    if args.gpu == True:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model_setup.load_model(args.checkpoint)
    infer(model.to(device), args.path, args.category_names, args.top_k)
    
    
def infer(model, path, category, top_k):
    from torchvision import transforms
    from PIL import Image
    import torch
    import json
    

   
    img = Image.open(path)
    test_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([.485, .456, .406],
                                                             [.229, .224, .225])])
    with torch.no_grad():  
        model.eval()
        inp_img = (test_transform(img).unsqueeze(0)).to('cuda:0')
        output = model(inp_img)
        
        prob = torch.exp(output)
        
        top_val, top_idx = prob.topk(top_k, dim=1)
        
    
    with open('cat_to_name.json') as f:
        category_data = json.load(f)
    
    category_inv = {category_data[i]: i for i in category_data}
    answers = []

    top_idx = (top_idx[0].to('cpu')).tolist()
    print(top_idx)
    for idx in top_idx:
         idx = str(idx)
         answers.append(category_data[idx])

    classes = ','.join(answers)
    print(classes)
    return top_val, answers
    
if __name__ == '__main__':
    main()
    
    
    
    