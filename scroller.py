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
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # Output: 32x128x128
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output: 64x64x64
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: 128x32x32
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Output: 256x16x16
            nn.ReLU(),
            nn.Flatten(),  # Flatten for linear layer input
            nn.Linear(256*16*16, 1024),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_log_var = nn.Linear(1024, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 1024)

        self.decoder = nn.Sequential(
            nn.Linear(1024, 256*16*16),
            nn.ReLU(),
            nn.Unflatten(1, (256, 16, 16)),  # Unflatten to 256x16x16 for conv transpose input
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Output: 128x32x32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: 64x64x64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output: 32x128x128
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # Output: 1x256x256
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



# Function to load the model
def load_model(path, device):
    model = VariationalAutoencoder(latent_dim=LATENT_DIM).to(device)
    model.load_state_dict(torch.load(path))
    return model

# Function to generate images with one varying latent variable
def generate_scrolling_images(model, num_images, folder_path, scroll_index, constant_values):
    os.makedirs(folder_path, exist_ok=True)

    for i in range(num_images):
        with torch.no_grad():
            # Create a latent vector with constant values
            latent_vector = torch.tensor(constant_values, dtype=torch.float32).unsqueeze(0).to(device)

            # Modify the latent variable that you want to scroll
            latent_vector[0, scroll_index] = torch.randn(1).item()

            # Decode the latent vector
            generated_image = model.decode(latent_vector).cpu()

            # Convert to PIL Image and save
            generated_image = generated_image.squeeze(0)
            generated_image = transforms.ToPILImage()(generated_image)
            generated_image.save(os.path.join(folder_path, f"generated_image_{i+1}.png"))

# Parameters
LATENT_DIM = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'variational_autoencoder.pth' 
vae_model = load_model(model_path, device)
vae_model.eval()

# Generate images
num_generated_images = 50
scroll_index = 0  # Index of the latent variable to scroll
constant_values = [0.0, 0, 0]  # Constants for other latent variables
generate_scrolling_images(vae_model, num_generated_images, 'generated_photos', scroll_index, constant_values)

print(f"Generated {num_generated_images} scrolling images in 'generated_photos' folder.")
