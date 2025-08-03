import torch
import os
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import OmegaConf
import argparse
import random
import numpy as np

def set_seed(seed=42):
    """Set seed for all random number generators"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class GatedConvFusion(nn.Module):
    """Gated convolutional fusion module"""
    def __init__(self, in_channels=32, kernel_size=3, delta_weight=0.2):
        super().__init__()
        padding = kernel_size // 2
        self.delta_conv = nn.Conv1d(in_channels, in_channels, kernel_size, padding=padding)
        self.gate_conv = nn.Sequential(
            nn.Conv1d(in_channels * 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.delta_weight = delta_weight  # Residual weight

    def forward(self, low_feat, high_feat):
        """Forward pass for fusion"""
        delta = self.delta_conv((high_feat - low_feat))  # [B, C, L]
        gate_input = torch.cat([low_feat, high_feat], dim=1)  # [B, 2*C, L]
        gate = self.gate_conv(gate_input)  # [B, C, L]
        fused = low_feat + gate * delta * self.delta_weight
        return fused

def load_data(sample_path, zcond_path, z100_path):
    """Load sample, zcond, and z100 data"""
    sample_data = torch.load(sample_path)
    zcond = torch.load(zcond_path)
    sample = sample_data.get("generated_z100_denorm")
    z100 = torch.load(z100_path)
    assert sample is not None and zcond is not None and z100 is not None
    return sample, zcond, z100

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Finetune GatedConvFusion Model")
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    # Path configuration (all set to empty)
    train_sample_path = ""
    train_zcond_path = ""
    test_sample_path = ""
    test_zcond_path = ""
    train_z100_path = ""
    test_z100_path = ""
    model_path = ""
    config_file = ""
    save_dir = ""
    os.makedirs(save_dir, exist_ok=True)

    # Load train and test data
    train_sample, train_zcond, train_z100 = load_data(train_sample_path, train_zcond_path, train_z100_path)
    test_sample, test_zcond, test_z100 = load_data(test_sample_path, test_zcond_path, test_z100_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load VAE decoder (for comparison and test inference)
    config = OmegaConf.load(config_file)
    autoencoder_args = config.autoencoderkl.params
    in_channels = train_zcond.shape[1]
    autoencoder_args['num_channels'] = [64,32]
    autoencoder_args['latent_channels'] = in_channels
    model = torch.load(model_path, map_location=device)
    if isinstance(model, dict) and 'model_state_dict' in model:
        from generative.networks.nets import AutoencoderKL
        vae = AutoencoderKL(**autoencoder_args).to(device)
        vae.load_state_dict(model['model_state_dict'])
    else:
        vae = model.to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # Directly output zcond through decoder and save (for comparison)
    with torch.no_grad():
        recon_zcond = vae.decode(test_zcond.to(device)).cpu()
    torch.save(recon_zcond, os.path.join(save_dir, "test_zcond_decoder_recon.pth"))
    print(f"Reconstructed data from test_zcond via decoder saved to {os.path.join(save_dir, 'test_zcond_decoder_recon.pth')}")

    # Construct fusion module
    fusion = GatedConvFusion(in_channels=in_channels, kernel_size=3, delta_weight=0.01).to(device)

    # Checkpoint path
    fusion_ckpt_path = os.path.join(save_dir, "fusion_model_checkpoint.pth")
    start_epoch = 0

    # Create train and test datasets
    train_dataset = TensorDataset(train_sample, train_zcond, train_z100)
    test_dataset = TensorDataset(test_sample, test_zcond, test_z100)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(fusion.parameters(), lr=1e-4)
    mse_loss = nn.MSELoss()

    # Check for existing checkpoint
    if os.path.exists(fusion_ckpt_path):
        print(f"Detected model checkpoint, resuming from: {fusion_ckpt_path}")
        checkpoint = torch.load(fusion_ckpt_path, map_location=device)
        fusion.load_state_dict(checkpoint['fusion_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resuming training from epoch {start_epoch}")

    num_epochs = args.num_epochs
    for epoch in range(start_epoch, num_epochs):
        fusion.train()
        epoch_loss = 0.0
        for sample, zcond, z100 in train_loader:
            sample = sample.to(device)
            zcond = zcond.to(device)
            z100 = z100.to(device)
            fused = fusion(zcond, sample)
            loss = mse_loss(fused, z100)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * sample.size(0)
        avg_loss = epoch_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Latent MSE: {avg_loss:.6f}")

        # Save checkpoint
        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            checkpoint = {
                'fusion_state_dict': fusion.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'train_loss': avg_loss
            }
            torch.save(checkpoint, fusion_ckpt_path)
            print(f"Checkpoint saved at epoch {epoch+1}")

        # Evaluate on test set each epoch
        fusion.eval()
        test_loss = 0.0
        with torch.no_grad():
            for sample, zcond, z100 in test_loader:
                sample = sample.to(device)
                zcond = zcond.to(device)
                z100 = z100.to(device)
                fused = fusion(zcond, sample)
                loss = mse_loss(fused, z100)
                test_loss += loss.item() * sample.size(0)
        avg_test_loss = test_loss / len(test_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - Test Latent MSE: {avg_test_loss:.6f}")

    # Save fine-tuned fusion module
    torch.save(fusion.state_dict(), os.path.join(save_dir, "gated_conv_fusion.pth"))
    print(f"Fusion module saved to {save_dir}")

    # Inference on test set and save reconstructed data
    fusion.eval()
    vae.eval()
    recon_list = []
    with torch.no_grad():
        for sample, zcond, z100 in test_loader:
            sample = sample.to(device)
            zcond = zcond.to(device)
            fused = fusion(zcond, sample)
            recon = vae.decode(fused)
            recon_list.append(recon.cpu())
    recon_all = torch.cat(recon_list, dim=0)
    torch.save(recon_all, os.path.join(save_dir, "test_reconstructed_data.pth"))
    print(f"Test set reconstructed data saved to {os.path.join(save_dir, 'test_reconstructed_data.pth')}")

if __name__ == "__main__":
    main()