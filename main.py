import argparse
from resnet.utils import SaveModel
from resnet.resnet import ResNetPretrained
from resnet.train import train_model
from resnet.data import get_dataloaders, generate_train_test_val

import torch
from torch import optim, nn
from torchsummary import summary
import numpy as np
import os


def setup():

    parser = argparse.ArgumentParser()
    parser.add_argument('-nc', '--num_classes', type=int, default=4, 
                        help='number of classes to train resnet')
    parser.add_argument('-tpre', '--tune_pretrained', type=bool, default=True, 
                        help='Tuning the pretrained model (optional)')
    parser.add_argument('-e', '--epochs', type=int, default=15, 
                        help='number of epochs to train resnet')
    parser.add_argument('-bs', '--batch_size', type=int, default=16, 
                        help='batch size of traning, test and valid stages')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, 
                        help='hyperparameter learning rate to define optimizer')
    parser.add_argument('-m', '--momentum', type=float, default=0.9, 
                        help='hyperparameter momentum to define optimizer')
    parser.add_argument('-sts', '--step_size', type=int, default=7, 
                        help='hyperparameter step size for scheduler')
    parser.add_argument('-g', '--gamma', type=float, default=0.1, 
                        help='hyperparameter gamma for scheduler')
    parser.add_argument('-cu', '--cuda', type=bool, default=True, 
                        help='training with cuda')
    parser.add_argument('-v', '--version', type=int, default=1, 
                        help='version of the model')
    args = vars(parser.parse_args())

    
    return args





def main(args):


    print(args)

    if args['cuda'] and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print(f"Computation device: {device}\n")

    data_path_init = os.path.join('data')
    data_path_init = {
        'train': os.path.join(data_path_init, 'Training'), 
        'test' : os.path.join(data_path_init, 'Testing')
        }
    data_paths = generate_train_test_val(data_path_init, 'tumor_dataset')
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    dataloaders = get_dataloaders(data_paths, mean, std, args['batch_size'])


    model = ResNetPretrained(num_classes=args['num_classes'], tune_pretrained=args['tune_pretrained'])
    model = model.to(device)
    summary(model, (3, 224, 224), device = device.type)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args['learning_rate'], momentum=args['momentum'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args['step_size'], gamma=args['gamma'])

    save_model = SaveModel('outputs')

    train_model(model, dataloaders, optimizer, scheduler, criterion, args['epochs'], device, save_model, args['version'])


if __name__ == '__main__':
    args = setup()
    main(args)

