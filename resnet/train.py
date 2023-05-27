from tqdm.auto import tqdm
import torch

def training_stage(model, trainloader, optimizer, criterion, device):

    print('Training Stage')

    model.train()
    
    train_epoch_loss = 0.0
    train_epoch_accuracy = 0
    counter = 0

    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)

        outputs = model(image)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        train_epoch_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        train_epoch_accuracy += (preds == labels).sum().item()
        
    epoch_loss = train_epoch_loss / counter
    epoch_acc = 100. * (train_epoch_accuracy / len(trainloader.dataset))
    return epoch_loss, epoch_acc

def validation_stage(model, valloader, criterion, device):

    print('Validation Stage')

    model.eval()
    
    valid_epoch_loss = 0.0
    valid_epoch_accuracy = 0
    counter = 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(valloader), total=len(valloader)):
            counter += 1
    
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            outputs = model(image)

            loss = criterion(outputs, labels)

            valid_epoch_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            valid_epoch_accuracy += (preds == labels).sum().item()

    epoch_loss = valid_epoch_loss / counter
    epoch_acc = 100. * (valid_epoch_accuracy / len(valloader.dataset))
    return epoch_loss, epoch_acc

def test_stage(model, testloader, criterion, device):

    print('Test Stage')

    model.eval()
    
    test_epoch_loss = 0.0
    test_epoch_accuracy = 0
    counter = 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
    
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            outputs = model(image)

            loss = criterion(outputs, labels)

            test_epoch_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            test_epoch_accuracy += (preds == labels).sum().item()

    epoch_loss = test_epoch_loss / counter
    epoch_acc = 100. * (test_epoch_accuracy / len(testloader.dataset))
    return epoch_loss, epoch_acc


def train_model(model, dataloaders, optimizer, scheduler, criterion, epochs, device, save_model, index):
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    
    for epoch in range(epochs):
            
            print(f"[INFO]: Epoch {epoch+1} of {epochs}")

            train_epoch_loss, train_epoch_acc = training_stage(model, dataloaders['train'], optimizer, criterion, device)
            valid_epoch_loss, valid_epoch_acc = validation_stage(model, dataloaders['val'], criterion, device)

            train_loss.append(train_epoch_loss)
            valid_loss.append(valid_epoch_loss)
            train_acc.append(train_epoch_acc)
            valid_acc.append(valid_epoch_acc)

            scheduler.step(valid_epoch_loss)

            print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
            print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
            
            save_model(valid_epoch_loss, valid_epoch_acc, epoch, model, optimizer, scheduler, dataloaders['val'], criterion, index)
            print('-'*50)

        
    save_model.save_model_final(epochs, model, optimizer, scheduler, dataloaders['train'], criterion, train_acc, valid_acc, train_loss, valid_loss, index)    
    save_model.save_plots(train_acc, valid_acc, train_loss, valid_loss, index)

    print('TRAINING COMPLETE')

    print('-'*37)
    print('TESTING THE MODEL')
    print('-'*37)

    test_epoch_loss, test_epoch_acc = test_stage(model, dataloaders['test'], criterion, device)
    print(f"Test loss: {test_epoch_loss} and Test accuracy: {test_epoch_acc}")