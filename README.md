# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset
The experiment aims to develop a binary classification model using a pretrained VGG19 to distinguish between defected and non-defected capacitors by modifying the last layer to a single neuron. The model will be trained on a dataset containing images of various defected and non-defected capacitors to enhance defect detection accuracy. Optimization techniques will be applied to improve performance, and the model will be evaluated to ensure reliable classification for capacitor quality assessment in manufacturing.


## DESIGN STEPS
### STEP 1: Problem Statement  
Define the objective of distinguishing between defected and non-defected capacitors using a binary classification model based on a pretrained VGG19.  

### STEP 2: Dataset Collection  
Use a dataset containing images of defected and non-defected capacitors for model training and evaluation.  

### STEP 3: Data Preprocessing  
Resize images to match VGG19 input dimensions, normalize pixel values, and create DataLoaders for efficient batch processing.  

### STEP 4: Model Architecture  
Modify the pretrained VGG19 by replacing the last layer with a single neuron using a sigmoid activation function for binary classification.  

### STEP 5: Model Training  
Train the model using a suitable loss function (Binary Cross-Entropy) and optimizer (Adam) for multiple epochs to enhance defect detection accuracy.  

### STEP 6: Model Evaluation  
Evaluate the model on unseen data using accuracy, precision, recall, and an ROC curve to assess classification performance.  

### STEP 7: Model Deployment & Visualization  
Save the trained model, visualize predictions, and integrate it into a manufacturing quality assessment system if needed.

## PROGRAM
Include your code here
```python
# Load Pretrained Model and Modify for Transfer Learning
model = models.vgg19(pretrained=True)


# Modify the final fully connected layer to match the dataset classes
in_features=model.classifier[-1].in_features
num_classes = len(train_dataset.classes)
model.classifier[-1] = nn.Linear(in_features, 1)


# Include the Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)



# Train the model
def train_model(model, train_loader, test_loader, num_epochs=50):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)  # Ensure proper shape for BCEWithLogitsLoss
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                labels = labels.float().unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name:Aadithya R      ")
    print("Register Number:212223240001      ")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()



```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot

![Screenshot 2025-04-16 193429](https://github.com/user-attachments/assets/b227f2d6-91b3-45a8-be8c-abe52b9cc9f8)
![Screenshot 2025-04-16 193454](https://github.com/user-attachments/assets/f50f0b7d-290b-4367-8af9-07d946ff8fd9)


### Confusion Matrix

![Screenshot 2025-04-16 193516](https://github.com/user-attachments/assets/2f3e2fc6-568e-4792-94da-c361dec8d20f)


### Classification Report

![Screenshot 2025-04-16 193532](https://github.com/user-attachments/assets/5dfe4f62-aa5c-4135-9b08-41de85d3632a)

### New Sample Prediction

![Screenshot 2025-04-16 193555](https://github.com/user-attachments/assets/dbe2278b-ee5e-4cce-b51d-108df79f8431)
![Screenshot 2025-04-16 193610](https://github.com/user-attachments/assets/40202c24-de82-42cf-9232-64d9e12c7a24)


## RESULT
The VGG-19 model was successfully trained and optimized to classify defected and non-defected capacitors.
