import argparse
from omegaconf import OmegaConf
import numpy as np
import torch
import random
import os
import time
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import MultiStepLR
from generative.losses import JukeboxLoss, PatchAdversarialLoss
from generative.networks.nets import AutoencoderKL, PatchDiscriminator
import torch.nn as nn
from torch import optim

# Import dataset functions (assuming they're in a separate module)
from dataset_module import load_SEED_data, load_DEAP_data_mean_std, DEAP, SEED_condition

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Autoencoder Training Script")
    parser.add_argument('--dataset_name', type=str, default='SEED', help='Dataset name')
    parser.add_argument('--patient_id', type=str, default='4', help='Patient ID')
    parser.add_argument('--shuffle', type=bool, default=True, help='Shuffle dataset')
    parser.add_argument('--num_channel', type=int, default=62, help='Number of EEG channels')
    parser.add_argument('--signal_length', type=int, default=800, help='Signal length')
    parser.add_argument('--dataset_dir', type=str, default='', help='Dataset directory')
    parser.add_argument('--save_dir', type=str, default='', help='Directory to save results')
    parser.add_argument('--down_stream_dir', type=str, default="", help='Downstream results directory')
    parser.add_argument('--epoch', type=int, default=200, help='Training epochs')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--mask_training', type=bool, default=True, help='Use masked training')
    parser.add_argument('--train_batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--network', type=str, default="CatConv_new", help='Network architecture')
    parser.add_argument('--num_timesteps', type=int, default=50, help='Number of timesteps')
    parser.add_argument('--noise', type=str, default="white", help='Noise type')
    parser.add_argument('--config_file', type=str, default="", help='Configuration file path')
    return parser.parse_args()

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def init_dataset(cfg):
    """Initialize dataset and data loaders"""
    # Load SEED dataset
    train_eeg, train_labels, train_mask, test_eeg, test_labels, test_mask = load_SEED_data(
        pat_id=cfg.patient_id,
        signal_length=cfg.signal_length,
        shuffle=cfg.shuffle,
        filepath=cfg.dataset_dir,
    )
    
    # Convert to PyTorch tensors
    train_eeg = torch.from_numpy(train_eeg).float()
    test_eeg = torch.from_numpy(test_eeg).float()
    train_labels = torch.from_numpy(train_labels)
    test_labels = torch.from_numpy(test_labels)
    
    # Create datasets
    train_dataset = TensorDataset(train_eeg, train_labels)
    test_dataset = TensorDataset(test_eeg, test_labels)
    
    # Create data loaders
    dataloader_kwargs = {"num_workers": 0}  # Simplified for portability
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        **dataloader_kwargs
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=False,
        **dataloader_kwargs
    )
    
    return train_loader, test_loader, train_eeg, test_eeg, train_labels, test_labels

def test_and_save_reconstruction(model, test_loader, cfg, device):
    """Test reconstruction quality and save results"""
    print("\nTesting reconstruction quality...")
    
    model.eval()
    original_list = []
    reconstructed_list = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            eeg_data = batch[0].to(device)
            
            # Forward pass
            reconstruction, miu, sigma = model(eeg_data)
            
            # Save results
            original_list.append(eeg_data.cpu())
            reconstructed_list.append(reconstruction.cpu())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed batch {batch_idx + 1}/{len(test_loader)}")
    
    # Concatenate all batches
    original_data = torch.cat(original_list, dim=0)
    reconstructed_data = torch.cat(reconstructed_list, dim=0)
    
    print(f"Original data shape: {original_data.shape}")
    print(f"Reconstructed data shape: {reconstructed_data.shape}")
    
    # Create save directory
    os.makedirs(cfg.down_stream_dir, exist_ok=True)
    
    # Save data
    original_path = os.path.join(cfg.down_stream_dir, "original_test_data.pth")
    reconstructed_path = os.path.join(cfg.down_stream_dir, "reconstructed_test_data.pth")
    
    torch.save(original_data, original_path)
    torch.save(reconstructed_data, reconstructed_path)
    
    print(f"Original test data saved to: {original_path}")
    print(f"Reconstructed test data saved to: {reconstructed_path}")
    
    # Calculate reconstruction metrics
    mse_loss = torch.mean((original_data - reconstructed_data) ** 2)
    mae_loss = torch.abs(original_data - reconstructed_data).mean()
    
    print(f"Reconstruction metrics:")
    print(f"  MSE Loss: {mse_loss:.6f}")
    print(f"  MAE Loss: {mae_loss:.6f}")
    
    # Per-channel metrics
    channel_mse = torch.mean((original_data - reconstructed_data) ** 2, dim=(0, 2))
    print(f"  Average channel MSE: {channel_mse.mean():.6f} ± {channel_mse.std():.6f}")
    
    return original_data, reconstructed_data

def single_inference_speed_test(model, test_loader, device):
    """Test inference speed on a single sample"""
    print("\nTesting inference speed...")
    
    model.eval()
    
    # Select random sample
    test_dataset = test_loader.dataset
    random_idx = torch.randint(0, len(test_dataset), (1,)).item()
    sample = test_dataset[random_idx]
    sample_eeg = sample[0].unsqueeze(0).to(device).float()
    
    print(f"Selected sample index: {random_idx}")
    print(f"Input shape: {sample_eeg.shape}")
    
    # Time inference
    start_time = time.time()
    with torch.no_grad():
        reconstruction, z_mu, z_sigma = model(sample_eeg)
    elapsed_time = time.time() - start_time
    
    print(f"Output shape: {reconstruction.shape}")
    print(f"Latent mean shape: {z_mu.shape}")
    print(f"Latent std shape: {z_sigma.shape}")
    print(f"Inference time: {elapsed_time:.6f} seconds")
    print(f"Inference speed: {1/elapsed_time:.2f} samples/second")
    
    return elapsed_time, reconstruction.cpu()

def main(cfg):
    """Main training function"""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config_file) if args.config_file else OmegaConf.create()
    
    # Update configuration with arguments
    cfg = OmegaConf.merge(cfg, OmegaConf.create(vars(args)))
    
    # Set random seed
    if cfg.seed is not None:
        set_seed(cfg.seed)
    
    # Set device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize dataset
    train_loader, test_loader, _, _, _, _ = init_dataset(cfg)
    print(f"Train samples: {len(train_loader.dataset)}, Test samples: {len(test_loader.dataset)}")
    
    # Initialize models
    autoencoder_args = config.get("autoencoderkl", {}).get("params", {})
    autoencoder_args.setdefault("num_channels", [32, 24, 16])
    autoencoder_args.setdefault("latent_channels", 8)
    
    model = AutoencoderKL(**autoencoder_args).to(device)
    
    discriminator_args = config.get("patchdiscriminator", {}).get("params", {})
    discriminator = PatchDiscriminator(**discriminator_args).to(device)
    
    # Initialize optimizers
    optimizer_g = optim.Adam(model.parameters(), lr=config.get("models", {}).get("optimizer_g_lr", 1e-4))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=config.get("models", {}).get("optimizer_d_lr", 1e-4))
    
    # Loss functions
    l1_loss = nn.L1Loss()
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    adv_weight = config.get("models", {}).get("adv_weight", 0.01)
    kl_weight = config.get("models", {}).get("kl_weight", 1e-6)
    spectral_weight = config.get("models", {}).get("spectral_weight", 0.1)
    
    # Training parameters
    n_epochs = config.get("train", {}).get("n_epochs", 100)
    val_interval = config.get("train", {}).get("val_interval", 10)
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(cfg.save_dir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load checkpoint if exists
    start_epoch = 0
    best_loss = float("inf")
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("epoch_") and f.endswith(".pth")]
    if checkpoint_files:
        latest_file = max(checkpoint_files, key=lambda f: int(f.split("_")[1].split(".")[0]))
        checkpoint_path = os.path.join(checkpoint_dir, latest_file)
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint.get('best_loss', float("inf"))
    
    # Training loop
    print(f"Starting training for {n_epochs} epochs...")
    for epoch in range(start_epoch, n_epochs):
        model.train()
        discriminator.train()
        
        epoch_recon_loss = 0
        epoch_gen_loss = 0
        epoch_disc_loss = 0
        num_batches = len(train_loader)
        
        print(f"\nEpoch {epoch+1}/{n_epochs}")
        print("-" * 50)
        
        for step, batch in enumerate(train_loader):
            eeg_data = batch[0].to(device)
            
            # Generator training
            optimizer_g.zero_grad()
            reconstruction, z_mu, z_sigma = model(eeg_data)
            
            # Reconstruction loss
            recons_loss = l1_loss(reconstruction, eeg_data)
            
            # KL divergence loss
            kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1)
            kl_loss = kl_loss / kl_loss.shape[0]
            
            # Adversarial loss
            logits_fake = discriminator(reconstruction.contiguous().float())[-1]
            generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            
            # Total generator loss
            loss_g = recons_loss + kl_weight * kl_loss + adv_weight * generator_loss
            loss_g.backward()
            optimizer_g.step()
            
            # Discriminator training
            optimizer_d.zero_grad()
            
            # Fake data
            logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
            loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
            
            # Real data
            logits_real = discriminator(eeg_data.contiguous().detach())[-1]
            loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
            
            # Total discriminator loss
            discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
            loss_d = adv_weight * discriminator_loss
            loss_d.backward()
            optimizer_d.step()
            
            # Update metrics
            epoch_recon_loss += recons_loss.item()
            epoch_gen_loss += generator_loss.item()
            epoch_disc_loss += discriminator_loss.item()
            
            # Print progress
            if (step + 1) % 10 == 0 or step == 0:
                print(f"Batch [{step+1}/{num_batches}] - "
                      f"Recon: {recons_loss.item():.6f}, "
                      f"Gen: {generator_loss.item():.6f}, "
                      f"Disc: {discriminator_loss.item():.6f}")
        
        # Calculate epoch averages
        avg_recon = epoch_recon_loss / num_batches
        avg_gen = epoch_gen_loss / num_batches
        avg_disc = epoch_disc_loss / num_batches
        
        print(f"Epoch Summary:")
        print(f"  Recon Loss: {avg_recon:.6f}")
        print(f"  Gen Loss: {avg_gen:.6f}")
        print(f"  Disc Loss: {avg_disc:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % val_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'best_loss': best_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    print("\nTraining completed!")
    
    # Save final model
    final_model_path = os.path.join(cfg.save_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved: {final_model_path}")
    
    # Test reconstruction quality
    print("\nTesting reconstruction quality...")
    original, reconstructed = test_and_save_reconstruction(model, test_loader, cfg, device)
    
    # Test inference speed
    print("\nTesting inference speed...")
    inference_time, _ = single_inference_speed_test(model, test_loader, device)
    print(f"Average inference time: {inference_time:.6f} seconds")
    
    print("\nAll tasks completed!")

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Create base configuration
    cfg = OmegaConf.create()
    
    # Run main function
    main(cfg)