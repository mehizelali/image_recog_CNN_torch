import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
from data.dataset import load_dataset





def train_eval_model(
        model,
        optimizer, 
        criterion,
        epochs,
        dataset_name,
        img_res,
        batch_size,
        device
    ):
    
    model.to(device)
    model.train()

    train_data, test_data, num_classes = load_dataset(name=dataset_name, download=True, img_res=img_res)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # Lists to store metrics
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        running_loss = 0
        correct = 0
        total = 0

        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total * 100

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {epoch_loss:.4f} - Train Accuracy: {epoch_accuracy:.4f}%")

        model.eval()

        test_running_loss = 0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                test_loss = criterion(outputs, labels)

                test_running_loss += test_loss.item()

                _, predicted_test = torch.max(outputs, 1)
                test_correct += (predicted_test == labels).sum().item()
                test_total += labels.size(0)

        epoch_test_loss = test_running_loss / len(test_loader)
        epoch_test_accuracy = test_correct / test_total * 100

        test_losses.append(epoch_test_loss)
        test_accuracies.append(epoch_test_accuracy)

        print(f"Epoch [{epoch+1}/{epochs}] - Test Loss: {epoch_test_loss:.4f} - Test Accuracy: {epoch_test_accuracy:.4f}%")

    return train_losses, train_accuracies, test_losses, test_accuracies











            

