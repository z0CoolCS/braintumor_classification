from torchvision import datasets
from torchvision import transforms

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import os
import shutil

def generate_train_test_val(data_paths, new_data_path, test_size = 0.1, val_size = 0.3):


    if os.path.isdir(new_data_path):
        shutil.rmtree(new_data_path)
        os.makedirs(new_data_path, exist_ok=True)

    new_train_path = os.path.join(new_data_path, 'train')
    new_test_path = os.path.join(new_data_path, 'test')
    new_val_path = os.path.join(new_data_path, 'val')

    os.makedirs(new_train_path, exist_ok=True)
    os.makedirs(new_test_path, exist_ok=True)
    os.makedirs(new_val_path, exist_ok=True)

    for class_name in os.listdir(data_paths['train']):

        os.makedirs(os.path.join(new_train_path, class_name), exist_ok=True)
        os.makedirs(os.path.join(new_test_path, class_name), exist_ok=True)
        os.makedirs(os.path.join(new_val_path, class_name), exist_ok=True)
        
        image_paths = []
        class_path_train = os.path.join(data_paths['train'], class_name)
        for filename in os.listdir(class_path_train):
            image_paths.append((class_path_train, filename))

        class_path_test = os.path.join(data_paths['test'], class_name)
        for filename in os.listdir(class_path_test):
            image_paths.append((class_path_test, filename))
        
        images_train, images_test = train_test_split(image_paths, test_size=test_size, shuffle=True, random_state=42)

        images_train_train, images_train_valid = train_test_split(images_train, test_size=val_size, shuffle=True, random_state=42)

        print(f"Images in Train:{len(images_train_train)}, Test:{len(images_test)}, Valid:{len(images_train_valid)} for class {class_name}")

        for img_path, filename_image in images_train_train:
            shutil.copyfile(
                  os.path.join(img_path, filename_image), 
                  os.path.join(new_train_path, class_name, filename_image)
                  )
                 
        for img_path, filename_image in images_test:
            shutil.copyfile(
                os.path.join(img_path, filename_image), 
                os.path.join(new_test_path, class_name, filename_image)
                )
                
        for img_path, filename_image in images_train_valid:
            shutil.copyfile(
                os.path.join(img_path, filename_image), 
                os.path.join(new_val_path, class_name, filename_image)
                )
            
    print("Train, test and valid dataset were generated successfully!")

    data_paths = {
        'train': new_train_path,
        'test': new_test_path,
        'val': new_val_path 
    }
    return data_paths

def shuffle_data(data_paths, test_size):
    for class_name in os.listdir(data_paths['train']):
            image_paths = []
            class_path_train = os.path.join(data_paths['train'], class_name)
            for filename in os.listdir(class_path_train):
                image_paths.append((class_path_train, filename))

            class_path_test = os.path.join(data_paths['test'], class_name)
            for filename in os.listdir(class_path_test):
                image_paths.append((class_path_test, filename))
            
            images_train, images_test = train_test_split(image_paths, test_size=test_size, shuffle=True, random_state=42)
            
            for img_path, filename_image in images_train:
                 shutil.move(
                      os.path.join(img_path, filename_image), 
                      os.path.join(class_path_train, filename_image)
                      )
                 
            for img_path, filename_image in images_test:
                 shutil.move(
                      os.path.join(img_path, filename_image), 
                      os.path.join(class_path_test, filename_image)
                      )


def get_dataloaders(data_paths, mean, std, batch_size):

            
    transforms = get_transforms(mean, std)

    image_train_dataset = datasets.ImageFolder(data_paths['train'], transform=transforms['train'])
    image_test_dataset = datasets.ImageFolder(data_paths['test'], transform=transforms['test'])
    image_val_dataset = datasets.ImageFolder(data_paths['val'], transform=transforms['val'])

    dataloaders = {}
    dataloaders['train'] = DataLoader(image_train_dataset, batch_size=batch_size, shuffle=True)
    dataloaders['test'] = DataLoader(image_test_dataset, batch_size=batch_size, shuffle=False)
    dataloaders['val'] = DataLoader(image_val_dataset, batch_size=batch_size, shuffle=True)

    return dataloaders


def get_transforms(mean, std):
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(45),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }
    
    return data_transforms

