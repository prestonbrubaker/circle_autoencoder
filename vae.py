import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

# Model Parameters
latent_dim = 3  # Example latent space dimension
LATENT_DIM = latent_dim

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=4, stride=2, padding=1),  # Output: 3x128x128
            nn.ReLU(),
            nn.Conv2d(3, 9, kernel_size=4, stride=2, padding=1),  # Output: 9x64x64
            nn.ReLU(),
            nn.Conv2d(9, 27, kernel_size=4, stride=2, padding=1),  # Output: 27x32x32
            nn.ReLU(),
            nn.Conv2d(27, 27, kernel_size=4, stride=2, padding=1),  # Output: 27x16x16
            nn.ReLU(),
            nn.Flatten(),  # Flatten for linear layer input
            nn.Linear(27*16*16, 256),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_log_var = nn.Linear(256, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 256)

        self.decoder = nn.Sequential(
            nn.Linear(256, 27*16*16),
            nn.ReLU(),
            nn.Unflatten(1, (27, 16, 16)),  # Unflatten to 256x16x16 for conv transpose input
            nn.ConvTranspose2d(27, 27, kernel_size=4, stride=2, padding=1),  # Output: 27x32x32
            nn.ReLU(),
            nn.ConvTranspose2d(27, 9, kernel_size=4, stride=2, padding=1),  # Output: 9x64x64
            nn.ReLU(),
            nn.ConvTranspose2d(9, 3, kernel_size=4, stride=2, padding=1),  # Output: 3x128x128
            nn.ReLU(),
            nn.ConvTranspose2d(3, 1, kernel_size=4, stride=2, padding=1),  # Output: 1x256x256
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

def loss_function(recon_x, x, mu, log_var):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD



# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        self.folder_path = folder_path
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.file_list[idx])
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        return image



# Load dataset


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


# Instantiate the dataset
dataset = CustomDataset(folder_path='photos', transform=transform)

# Dataset and Dataloader
dataloader = DataLoader(dataset, batch_size=24, shuffle=True)

# Instantiate VAE model with latent_dim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using: " + str(device))
model = VariationalAutoencoder(latent_dim=LATENT_DIM).to(device)

# Loss and optimizer
optimizer = optim.Adam(model.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.00, amsgrad=False)

# Train the model
num_epochs = 100000
for epoch in range(num_epochs):
    for data in dataloader:
        img = data.to(device)

        # Forward pass
        recon_batch, mu, log_var = model(img)

        # Calculate loss
        loss = loss_function(recon_batch, img, mu, log_var)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
        optimizer.step()

    if epoch % 1 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')
        torch.save(model.state_dict(), f'variational_autoencoder.pth')
        print("Model Saved at Epoch: ", epoch)

# Save the final model
torch.save(model.state_dict(), 'variational_autoencoder_final.pth')
