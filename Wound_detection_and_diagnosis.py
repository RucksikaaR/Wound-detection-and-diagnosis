import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, utils
from PIL import Image
import numpy as np
data = {'Path': [],'Class': []}
dataset_path = r'the path of the location where your dataset is saved'
entries = os.listdir(dataset_path)

#To convert the string labels to numerical ones as it can make the concatenation of tensors easier
label_mapping = {'Abrasions': 0,
                 'Bruises': 1,
                 'Burns': 2,
                 'Cut': 3,
                 'Diabetic Wounds': 4,
                 'Laceration': 5,
                 'Normal': 6,
                 'Pressure Wounds': 7,
                 'Surgical Wounds': 8,
                 'Venous Wounds': 9}

for entry in entries:
    full_path = os.path.join(dataset_path,entry)
    if os.path.isdir(full_path):
        files = os.listdir(full_path)
        for file in files:
            file_path = os.path.join(full_path,file)
            data['Path'].append(os.path.join(entry,file))
            data['Class'].append(entry)

df = pd.DataFrame(data)

# Custom Dataset Class
class CustomImageDataset(Dataset):
    def __init__(self, dataframe, indices= None, transform=None):
        self.dataframe = dataframe
        self.indices = indices if indices is not None else range(len(dataframe))
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self,idx):
        img_idx = self.indices[idx]
        img_path = self.dataframe.iloc[img_idx, 0]
        image = Image.open(img_path).convert("RGB")
        label = self.dataframe.iloc[img_idx, 1]

    if self.transform:
        image = self.transform(image)

    return image, label

# Define Transformations
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) #Normalizing the image using ImageNet stats

# Split the dataset into training, validation, and test sets
train_size = int(0.7*len(df))
val_size = int(0.15*len(df))
test_size = len(df) - train_size - val_size

# random_split returns Subset objects containing indices
train_subset, val_subset, test_subset = random_split(df, [train_size, val_size, test_size])
train_df=df.iloc[train_subset.indices]
minority_class_df = train_df[train_df['Class'] == 2]

# Extract indices from the subsets
train_indices = train_subset.indices
val_indices = val_subset.indices
test_indices = test_subset.indices
minority_indices = minority_class_df.index

# Create Datasets and DataLoaders
train_dataset = CustomImageDataset(dataframe=df, indices=train_indices, transform=transform)
val_dataset = CustomImageDataset(dataframe=df, indices=val_indices, transform=transform)
test_dataset = CustomImageDataset(dataframe=df, indices=test_indices, transform=transform)
minority_class_dataset = CustomImageDataset(dataframe=minority_class_df, indices = minority_indices, transform=transform)

train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=32,shuffle=False)
test_loader = DataLoader(test_dataset,batch_size=32,shuffle=False)

# Define the GAN
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(5, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256,512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 3 * 128 * 128),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 3, 128, 128)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3 * 128 * 128, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

#Initialize the models
generator = Generator()
discriminator = Discriminator()

# Device configuration and move models to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #torch.cuda.is_available() will return true if build was successful in ROCm software and your AMD GPU will be detected as 'cuda'
generator = generator.to(device)
discriminator = discriminator.to(device)

# Loss function and optimizers
criterion_GAN = nn.BCELoss()
generator_optimizer = optim.Adam(generator.parameters(), lr=0.0005, betas=(0.5, 0.999), weight_decay=1e-4)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999), weight_decay=1e-4)

# Early Stopping Parameters
patience = 10
best_generator_loss = float('inf')
best_discriminator_loss = float('inf')
epochs_no_improve = 0

#Train the GAN
def train_gan(generator, discriminator, dataloader, epochs=500, display_interval=2, patience=10):
    global best_generator_loss, best_discriminator_loss, epochs_no_improve
    for epoch in range(epochs):
        for i, data in enumerate(dataloader):
            real_images,_ = data #real images and their corresponding labels. _ is used as labels are irrelevant in this case
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # Ground truths
            real_labels = torch.ones(batch_size, 1, requires_grad=False, device=device)
            fake_labels = torch.zeros(batch_size, 1, requires_grad=False, device=device)

            # Train Discriminator with real images
            discriminator_optimizer.zero_grad()
            real_outputs = discriminator(real_images)
            real_loss = criterion_GAN(real_outputs, real_labels)

            #Train discriminator with fake images
            noise = torch.randn(batch_size, 5, device=device) #Generate random noise vectors
            fake_images = generator(noise) #Generate fake images
            fake_outputs = discriminator(fake_images.detach())
            fake_loss = criterion_GAN(fake_outputs, fake_labels)
            discriminator_loss = (real_loss + fake_loss) / 2
            discriminator_loss.backward()
            discriminator_optimizer.step()

            # Train Generator
            generator_optimizer.zero_grad()
            fake_outputs = discriminator(fake_images)
            generator_loss = criterion_GAN(fake_outputs, real_labels)
            generator_loss.backward()
            generator_optimizer.step()

        # Early Stopping Check
        if generator_loss.item() < best_generator_loss and discriminator_loss.item() < best_discriminator_loss:
            best_generator_loss = generator_loss.item()
            best_discriminator_loss = discriminator_loss.item()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            print(f"[Epoch {epoch}/{epochs}] [Discriminator loss: {discriminator_loss.item()}] [Generator loss: {generator_loss.item()}]")
            break

        if epoch % display_interval == 0:
            print(f"[Epoch {epoch}/{epochs}] [Discriminator loss: {discriminator_loss.item()}] [Generator loss: {generator_loss.item()}]")

            # Visualize a few generated images
            with torch.no_grad():
                sample_noise = torch.randn(16, 5)
                generated_images = generator(sample_noise).cpu() #Moved from gpu to cpu to leverage computational power
                grid = utils.make_grid(generated_images, nrow=4, normalize=True)
                plt.figure(figsize=(8, 8))
                plt.imshow(grid.permute(1, 2, 0).numpy())
                plt.title(f"Generated Images at Epoch {epoch}")
                plt.show()

train_gan(generator, discriminator, train_loader)

# Augment the Dataset
def augment_data(generator, original_data,augment_size=100):
    noise = torch.randn(augment_size, 5)
    generated_images = generator(noise).detach().cpu()
    augmented_data = torch.cat((original_data, generated_images))
    return augmented_data, generated_images

# Extract original data
original_images = torch.cat([data[0] for data in train_loader])
original_labels = torch.cat([data[1] for data in train_loader])
augmented_x_train, generated_images = augment_data(generator, original_images)
generated_labels = torch.full((100,), 2, dtype=torch.long)

#Combine Original and Augmented dataset
augmented_labels = torch.cat((original_labels, generated_labels),dim=0)
augmented_trainset = torch.utils.data.TensorDataset(augmented_x_train, augmented_labels)
augmented_trainloader = DataLoader(augmented_trainset, batch_size=32, shuffle=True)

#Define the classifier
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 32 * 32, 512) #64 channels
        self.fc2 = nn.Linear(512, 10) #10 classes
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

#Initialize the Classifier
Classifier = Classifier()

#Loss function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer_C = optim.Adam(Classifier.parameters(), lr=0.0001)

#Evaluate the Classifier
def evaluate_classifier(model, val_loader,criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()*images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = val_loss/len(val_loader.dataset)
    val_accuracy = 100*correct/total
    return val_loss, val_accuracy

#Train the Classifier
def train_classifier(model,train_loader,val_loader,criterion,optimizer,epochs=10,patience=5):
    train_losses, val_losses, val_accuracies = [], [], []
    best_val_loss = float('inf')
    patience_counter=0

    for epoch in range(epochs):
        model.train()
        running_loss=0.0

    for images, labels in train_loader:
        images, labels = imgs.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    val_loss, val_accuracy = evaluate_classifier(model, val_loader, criterion)
    train_losses.append(epoch_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f"[Epoch {epoch+1}/{epochs}] [Loss: {epoch_loss:.4f}] [Val Loss: {val_loss:.4f}] [Val Accuracy: {val_accuracy:.2f}%]")
    if val_loss<best_val_loss:
        best_val_loss = val_loss
        patience_counter=0
    else:
        patience_counter+=1
    if patience_counter>=patience:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break
    return train_losses, val_losses, val_accuracies

# Train and Validate the Classifier
train_losses, val_losses, val_accuracies = train_classifier(Classifier, augmented_trainloader, val_loader, criterion, optimizer_C, epochs=10)

# Evaluate on the Test Set
test_loss, test_accuracy = evaluate_classifier(Classifier, test_loader, criterion)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

# Plotting the results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

#To predict given image
def predict_unlabeled_image(model, image_path, transform):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        return predicted.item()

#To provide remarks 
def final_result(predicted):
    if predicted == 0:
        return "Abrasion. Maintain proper wound hygiene and control blood sugar if patient has diabetes. Get tetanus shot if wound was caused by dirty or rusty object, and if you hadn't had a shot in the last 5 years."
    elif predicted == 1:
        return "Bruise. If hematoma detected, please prepare to drain it. Consider physical or compression therapy to improve blood supply, and low-level laser therapy(LLLT) to help reduce inflammation and promote healing."
    elif predicted == 2:
        return "Burns. Monitor wound over the next 3 weeks and look for signs of healing. Prescribe antimicrobial ointments or silver-containing dressings to prevent infection. Consider Hyperbaric Oxygen Therapy(HBOT) or application of bioengineered skin substitutes to stimulate healing."
    elif predicted == 3:
        return "Cut. Get tetanus shot if wound was caused by dirty or rusty object, and if you hadn't had a shot in the last 5 years. Monitor wound if deep. Prescribe antimicrobial ointments or silver-containing dressings to prevent infection. Consider Negative Pressure Wound Therapy(NPWT) or application of bioengineered skin substitutes to stimulate healing."
    elif predicted == 4:
        return "Diabetic Ulcer. Monitor blood sugar levels and control sugar intake. Consider Hyperbaric Oxygen Therapy(HBOT) to promote healing. Minimize pressure on the ulcerated area using methods like total contact casting, removable cast walkers or specialized diabetic footwear. Advice patient to obtain plenty of rest and limit activities that pressure the ulcer."
    elif predicted == 5:
        return "Laceration. Get tetanus shot if wound was caused by dirty or rusty object, and if you hadn't had a shot in the last 5 years. Avoid further trauma and apply appropriate dressing(hydrocolloid, hydrogel, alginate or foam dressings) to keep wound clean and moist. Consider Hyperbaric Oxygen Therapy(HBOT), Negative Pressure Wound Therapy(NPWT) or application of bioengineered skin substitutes to promote healing."
    elif predicted == 6:
        return "Normal. The skin looks normal. Nothing to worry about."
    elif predicted == 7:
        return "Pressure Ulcer. Use specialized mattresses, cushions and pads to reduce pressure exerted on the patient and minimize the risk of new ulcers. Consider Hyperbaric Oxygen Therapy(HBOT)or Negative Pressure Wound Therapy(NPWT) to promote healing. Practice good hygiene and maintain healthy body weight. Advice patient to avoid smoking if patient smokes."
    elif predicted == 8:
        return "Surgical Wound. If patient is diabetic, monitor sugar levels and control sugar intake. Practice physical therapy or meditation to improve blood flow to the affected area. Prescribe antimicrobial ointments or silver-containing dressings to prevent infection."
    elif predicted == 9:
        return "Venous Ulcer. Apply compression therapy to reduce venous hypertension and for more effective pressure distribution. Encourage activities like walking and leg exercise and elevate the leg to reduce swelling and venous pressure."

#Apply the model to an undiagnosed wound
Image_Path = r'the path of the location where your dataset is saved'
wound_image=predict_unlabeled_image(Classifier,Image_path)
Diagnosis = final_result(wound_image)
print(Diagnosis)
