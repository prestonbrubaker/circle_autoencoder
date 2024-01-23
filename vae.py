import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


# Model Parameters
latent_dim = 3  # Example latent space dimension
LATENT_DIM = latent_dim

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            # Flatten the convolutional layer output
            nn.Flatten(),
            nn.Linear(256*16*16, 1024),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_log_var = nn.Linear(1024, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 1024)

        self.decoder = nn.Sequential(
            nn.Linear(1024, 256*16*16),
            nn.ReLU(),
            # Unflatten the layer to 256 channels with 16x16 image size
            nn.Unflatten(1, (256, 16, 16)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 256x256
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 32, 256, 256)
        return self.decoder(x)

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
dataloader = DataLoader(dataset, batch_size=600, shuffle=True)

# Instantiate VAE model with latent_dim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VariationalAutoencoder(latent_dim=LATENT_DIM).to(device)

# Loss and optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.00, amsgrad=False)

# Train the model
num_epochs = 100000
for epoch in range(num_epochs):
    for data in dataloader:
        img = data.to(device)

        # Forward pass
        recon_batch, mu, log_var = model(img)

        # Clamp the output of the model to ensure it's within [0, 1]
        recon_batch = torch.clamp(recon_batch, 0, 1)

        # Calculate loss
        loss = loss_function(recon_batch, img, mu, log_var)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 25 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')
        torch.save(model.state_dict(), f'variational_autoencoder.pth')
        print("Model Saved at Epoch: ", epoch)

# Save the final model
torch.save(model.state_dict(), 'variational_autoencoder_final.pth')
