import argparse
from omegaconf import OmegaConf
import numpy as np
import torch
import random
import os
import time
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import MultiStepLR
from generative.networks.schedulers import DDIMScheduler
from generative.networks.nets import UNetModel
from torch.nn import MSELoss

# Import custom modules
from network import AdaConv_Res_Small
from res_time_net import TimeAwareResidualPredictor
from train_conv_new import ConvEncoder

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Latent DDIM Training")
    parser.add_argument('--latent_data_dir', type=str, default='', help='Directory for latent data')
    parser.add_argument('--save_dir', type=str, default='', help='Directory to save checkpoints')
    parser.add_argument('--down_stream_dir', type=str, default='', help='Directory for downstream results')
    parser.add_argument('--epoch', type=int, default=80, help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--train_batch_size', type=int, default=64 * 4, help='Training batch size')
    parser.add_argument('--num_train_timesteps', type=int, default=1000, help='Number of training timesteps')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--config_file', type=str, default='', help='VAE configuration file')
    parser.add_argument('--conv_encoder_ckpt', type=str, default='', help='ConvEncoder checkpoint path')
    parser.add_argument('--conv_encoder_time_emb_dim', type=int, default=64, help='ConvEncoder time embedding dimension')
    parser.add_argument('--conv_encoder_T', type=int, default=1000, help='ConvEncoder T parameter')
    parser.add_argument('--scheduler_s', type=float, default=0.02, help='Scheduler s parameter')
    parser.add_argument('--scheduler_schedule', type=str, default='cosine', help='Scheduler schedule type')
    return parser.parse_args()

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
def train_residual_model(res_model, train_loader, optimizer, device, num_epochs, scheduler_steps):
    """Train residual prediction model"""
    res_model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader):
            z100_target = batch[0].to(device)
            z50 = batch[1].to(device)
            batch_size = z100_target.shape[0]
            timesteps = torch.randint(0, scheduler_steps, (batch_size,), device=device).long()
            res_target = z100_target - z50
            optimizer.zero_grad()
            res_pred = res_model(z50, timesteps)
            res_loss = torch.nn.functional.l1_loss(res_pred, res_target)
            res_loss.backward()
            optimizer.step()
            epoch_loss += res_loss.item()
        avg_loss = epoch_loss / len(train_loader)
        print(f"[Residual] Epoch {epoch+1}/{num_epochs} - Avg Res Loss: {avg_loss:.6f}")
    return res_model

def load_conv_encoder(conv_encoder_ckpt, in_channels, latent_channels, T, time_emb_dim, device):
    """Load pretrained ConvEncoder"""
    conv_encoder = ConvEncoder(
        in_channels=in_channels,
        latent_channels=latent_channels,
        T=T,
        time_emb_dim=time_emb_dim
    ).to(device)
    conv_encoder.load_state_dict(torch.load(conv_encoder_ckpt, map_location=device))
    conv_encoder.eval()
    for p in conv_encoder.parameters():
        p.requires_grad = False
    return conv_encoder

def train_latent_ddim(model, scheduler, train_loader, optimizer, device, epoch, cfg, conv_encoder):
    """Train latent DDIM model"""
    model.train()
    epoch_loss = 0.0
    num_batches = len(train_loader)
    for step, batch in enumerate(train_loader):
        z100_target = batch[0].to(device)
        z50 = batch[1].to(device)
        batch_size = z100_target.shape[0]
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (batch_size,), device=device).long()
        # Generate conditioning with ConvEncoder
        zcond = conv_encoder(z50, timesteps)
        noise = torch.randn_like(z100_target)
        noisy_z100 = scheduler.add_noise(z100_target, noise, timesteps)
        optimizer.zero_grad()
        noise_pred = model(noisy_z100, timesteps, zcond)
        loss = torch.nn.functional.smooth_l1_loss(noise_pred, noise)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
        if (step + 1) % 10 == 0 or step == 0:
            avg_loss = epoch_loss / (step + 1)
            print(f"Batch [{step+1}/{num_batches}] - Loss: {loss.item():.6f}, Avg Loss: {avg_loss:.6f}")
    return epoch_loss / num_batches

def sample_latent_ddim(model, scheduler, z50, conv_encoder, device, num_inference_steps=50):
    """Sample from latent DDIM model"""
    model.eval()
    with torch.no_grad():
        scheduler.set_timesteps(num_inference_steps)
        batch_size = z50.shape[0]
        z100_sample = torch.randn_like(z50)  # Initialize with same shape as z50
        for t in scheduler.timesteps:
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            # Generate conditioning for current timestep
            zcond = conv_encoder(z50, t_batch)
            noise_pred = model(z100_sample, t_batch, zcond)
            scheduler_output = scheduler.step(noise_pred, t, z100_sample)
            if hasattr(scheduler_output, 'prev_sample'):
                z100_sample = scheduler_output.prev_sample
            elif isinstance(scheduler_output, tuple):
                z100_sample = scheduler_output[0]
            else:
                z100_sample = scheduler_output
    return z100_sample

def load_latent_data(latent_data_dir):
    """Load latent space data and normalize"""
    print(f"Loading latent space data from: {latent_data_dir}")
    
    # Find data files
    train_z100_files = [f for f in os.listdir(latent_data_dir) if f.startswith('train_z100') and f.endswith('.pth')]
    train_z50_files = [f for f in os.listdir(latent_data_dir) if f.startswith('train_z50') and f.endswith('.pth')]
    test_z100_files = [f for f in os.listdir(latent_data_dir) if f.startswith('test_z100') and f.endswith('.pth')]
    test_z50_files = [f for f in os.listdir(latent_data_dir) if f.startswith('test_z50') and f.endswith('.pth')]
    
    if not train_z100_files or not train_z50_files:
        raise FileNotFoundError(f"Train data not found in {latent_data_dir}")
    if not test_z100_files or not test_z50_files:
        raise FileNotFoundError(f"Test data not found in {latent_data_dir}")
    
    # Use latest files
    train_z100_file = sorted(train_z100_files)[-1]
    train_z50_file = sorted(train_z50_files)[-1]
    test_z100_file = sorted(test_z100_files)[-1]
    test_z50_file = sorted(test_z50_files)[-1]
    
    # Load data
    train_z100_data = torch.load(os.path.join(latent_data_dir, train_z100_file), map_location='cpu')
    train_z50_data = torch.load(os.path.join(latent_data_dir, train_z50_file), map_location='cpu')
    test_z100_data = torch.load(os.path.join(latent_data_dir, test_z100_file), map_location='cpu')
    test_z50_data = torch.load(os.path.join(latent_data_dir, test_z50_file), map_location='cpu')
    
    print(f"Train z100 shape: {train_z100_data.shape}")
    print(f"Train z50 shape: {train_z50_data.shape}")
    print(f"Test z100 shape: {test_z100_data.shape}")
    print(f"Test z50 shape: {test_z50_data.shape}")
    
    # Check shape consistency
    if train_z100_data.shape != train_z50_data.shape:
        raise ValueError(f"Train z100 and z50 shape mismatch: {train_z100_data.shape} vs {train_z50_data.shape}")
    if test_z100_data.shape != test_z50_data.shape:
        raise ValueError(f"Test z100 and z50 shape mismatch: {test_z100_data.shape} vs {test_z50_data.shape}")
    
    # Combine for statistics
    all_z100_data = torch.cat([train_z100_data, test_z100_data], dim=0)
    all_z50_data = torch.cat([train_z50_data, test_z50_data], dim=0)
    
    # Calculate normalization parameters
    if len(all_z50_data.shape) == 4:  # [N, C, H, W]
        z50_mean = all_z50_data.mean(dim=(0, 2, 3), keepdim=True)
        z50_std = all_z50_data.std(dim=(0, 2, 3), keepdim=True)
    elif len(all_z50_data.shape) == 3:  # [N, C, L]
        z50_mean = all_z50_data.mean(dim=(0, 2), keepdim=True)
        z50_std = all_z50_data.std(dim=(0, 2), keepdim=True)
    else:
        raise ValueError(f"Unknown latent shape: {all_z50_data.shape}")
    
    # Avoid division by zero
    z50_std = torch.clamp(z50_std, min=1e-8)
    
    # Normalize data
    train_z50_normalized = (train_z50_data - z50_mean) / z50_std
    train_z100_normalized = (train_z100_data - z50_mean) / z50_std
    test_z50_normalized = (test_z50_data - z50_mean) / z50_std
    test_z100_normalized = (test_z100_data - z50_mean) / z50_std
    
    return (
        train_z100_normalized.float(),
        train_z50_normalized.float(),
        test_z100_normalized.float(),
        test_z50_normalized.float(),
        z50_mean.float(),
        z50_std.float()
    )

def create_data_loaders(train_z100_data, train_z50_data, test_z100_data, test_z50_data, batch_size):
    """Create data loaders for training and testing"""
    train_dataset = TensorDataset(train_z100_data, train_z50_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataset = TensorDataset(test_z100_data, test_z50_data)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    return train_loader, test_loader

def denormalize_data(normalized_data, mean, std):
    """Denormalize data using mean and std"""
    return normalized_data * std + mean

def test_sampling(model, scheduler, test_loader, device, cfg, conv_encoder, num_inference_steps=50, z50_mean=None, z50_std=None):
    """Test sampling performance"""
    print(f"\nTesting DDIM sampling (inference steps: {num_inference_steps})")
    
    model.eval()
    test_results = {
        'original_z100': [],
        'condition_z50': [],
        'generated_z100': [],
        'original_z100_denorm': [],
        'condition_z50_denorm': [],
        'generated_z100_denorm': [],
        'mse_scores': [],
        'mse_scores_denorm': [],
        'zcond': []  # Save conditioning vectors
    }
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            z100_target = batch[0].to(device)
            z50_condition = batch[1].to(device)
            batch_size = z100_target.shape[0]
            # Generate conditioning with t=0
            t0 = torch.zeros(batch_size, device=device, dtype=torch.long)
            zcond = conv_encoder(z50_condition, t0)
            test_results['zcond'].append(zcond.cpu())
            
            # Generate samples
            z100_generated = sample_latent_ddim(
                model, scheduler, z50_condition, conv_encoder, device, num_inference_steps
            )
            
            # Calculate MSE
            mse = torch.mean((z100_generated - z100_target) ** 2).item()
            
            # Denormalize if parameters provided
            if z50_mean is not None and z50_std is not None:
                z50_mean_device = z50_mean.to(device)
                z50_std_device = z50_std.to(device)
                z100_target_denorm = denormalize_data(z100_target, z50_mean_device, z50_std_device)
                z50_condition_denorm = denormalize_data(z50_condition, z50_mean_device, z50_std_device)
                z100_generated_denorm = denormalize_data(z100_generated, z50_mean_device, z50_std_device)
                mse_denorm = torch.mean((z100_generated_denorm - z100_target_denorm) ** 2).item()
                test_results['original_z100_denorm'].append(z100_target_denorm.cpu())
                test_results['condition_z50_denorm'].append(z50_condition_denorm.cpu())
                test_results['generated_z100_denorm'].append(z100_generated_denorm.cpu())
                test_results['mse_scores_denorm'].append(mse_denorm)
            else:
                mse_denorm = None
            
            # Save results
            test_results['original_z100'].append(z100_target.cpu())
            test_results['condition_z50'].append(z50_condition.cpu())
            test_results['generated_z100'].append(z100_generated.cpu())
            test_results['mse_scores'].append(mse)
            
            print(f"  Batch {batch_idx+1}: MSE = {mse:.6f}" + 
                  (f", Denorm MSE = {mse_denorm:.6f}" if mse_denorm is not None else ""))
    
    # Concatenate results
    for key in ['original_z100', 'condition_z50', 'generated_z100', 'zcond']:
        test_results[key] = torch.cat(test_results[key], dim=0)
    
    if z50_mean is not None and z50_std is not None:
        for key in ['original_z100_denorm', 'condition_z50_denorm', 'generated_z100_denorm']:
            test_results[key] = torch.cat(test_results[key], dim=0)
        avg_mse_denorm = np.mean(test_results['mse_scores_denorm'])
        print(f"Average sampling MSE (denormalized): {avg_mse_denorm:.6f}")
    
    avg_mse = np.mean(test_results['mse_scores'])
    print(f"Average sampling MSE (normalized): {avg_mse:.6f}")
    
    # Save normalization parameters
    test_results['z50_mean'] = z50_mean
    test_results['z50_std'] = z50_std
    
    return test_results, avg_mse

def main():
    """Main training function"""
    args = parse_args()
    
    # Set random seed
    if args.seed is not None:
        set_seed(args.seed)

    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.down_stream_dir, exist_ok=True)

    # Load data
    train_z100_data, train_z50_data, test_z100_data, test_z50_data, z50_mean, z50_std = load_latent_data(args.latent_data_dir)
    train_loader, test_loader = create_data_loaders(
        train_z100_data, train_z50_data, test_z100_data, test_z50_data, args.train_batch_size
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load ConvEncoder
    conv_encoder = load_conv_encoder(
        conv_encoder_ckpt=args.conv_encoder_ckpt,
        in_channels=train_z50_data.shape[1],
        latent_channels=train_z100_data.shape[1],
        T=args.conv_encoder_T,
        time_emb_dim=args.conv_encoder_time_emb_dim,
        device=device
    )

    # Create main model
    model = AdaConv_Res_Small(
        signal_length=train_z100_data.shape[-1],
        signal_channel=train_z100_data.shape[-2],
        hidden_channel=12,
        in_kernel_size=7,
        out_kernel_size=7,
        slconv_kernel_size=17,
        num_scales=4,
        num_blocks=4,
        num_off_diag=8,
        use_pos_emb=True,
        padding_mode="circular",
        use_fft_conv=True,
    ).to(device)

    # Create residual model
    res_model = TimeAwareResidualPredictor(
        channel_dim=train_z100_data.shape[-2],
        signal_length=train_z100_data.shape[-1],
        time_embed_dim=128,
        max_timesteps=args.num_train_timesteps,
        num_layers=2
    ).to(device)

    # Create scheduler
    scheduler = DDIMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        schedule=args.scheduler_schedule,
        s=args.scheduler_s,
        clip_sample=False,
        steps_offset=1
    )

    # Create optimizers
    res_optimizer = torch.optim.Adam(res_model.parameters(), lr=args.learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_scheduler = MultiStepLR(optimizer, milestones=[100, 150], gamma=0.5)

    # Train residual model
    res_checkpoint_path = os.path.join(args.save_dir, "res_model_checkpoint.pth")
    if os.path.exists(res_checkpoint_path):
        print(f"Loading residual model checkpoint: {res_checkpoint_path}")
        res_checkpoint = torch.load(res_checkpoint_path, map_location=device)
        res_model.load_state_dict(res_checkpoint['model_state_dict'])
        res_optimizer.load_state_dict(res_checkpoint['optimizer_state_dict'])
    else:
        print("\n=== Training residual prediction model ===")
        train_residual_model(
            res_model, train_loader, res_optimizer, device, num_epochs=500, scheduler_steps=args.num_train_timesteps
        )
        torch.save(res_model.state_dict(), os.path.join(args.save_dir, "res_model_pretrained.pth"))
        res_checkpoint = {
            'model_state_dict': res_model.state_dict(),
            'optimizer_state_dict': res_optimizer.state_dict(),
            'epoch': 500,
        }
        torch.save(res_checkpoint, res_checkpoint_path)

    # Main training loop
    start_epoch = 0
    best_train_loss = float("inf")
    train_losses = []
    
    # Check for existing checkpoint
    checkpoint_files = [f for f in os.listdir(args.save_dir) if f.startswith("checkpoint_epoch_") and f.endswith(".pth")]
    if checkpoint_files:
        latest_checkpoint_file = max(checkpoint_files, key=lambda f: int(f.split("_")[-1].split(".")[0]))
        checkpoint_path = os.path.join(args.save_dir, latest_checkpoint_file)
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_train_loss = checkpoint.get('best_train_loss', float("inf"))
        train_losses = checkpoint.get('train_losses', [])
    
    # Training loop
    for epoch in range(start_epoch, args.epoch):
        model.train()
        res_model.eval()  # Freeze residual model
        epoch_loss = 0.0
        num_batches = len(train_loader)
        
        for step, batch in enumerate(train_loader):
            z100_target = batch[0].to(device)
            z50 = batch[1].to(device)
            batch_size = z100_target.shape[0]
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (batch_size,), device=device).long()
            zcond = conv_encoder(z50, timesteps)
            noise = torch.randn_like(z100_target)
            noisy_z100 = scheduler.add_noise(z100_target, noise, timesteps)

            optimizer.zero_grad()
            noise_pred = model(noisy_z100, timesteps, zcond)
            ddpm_loss = torch.nn.functional.smooth_l1_loss(noise_pred, noise)
            ddpm_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += ddpm_loss.item()

            if (step + 1) % 10 == 0 or step == 0:
                avg_loss = epoch_loss / (step + 1)
                print(f"Batch [{step+1}/{num_batches}] - Loss: {ddpm_loss.item():.6f}, Avg Loss: {avg_loss:.6f}")

        avg_epoch_loss = epoch_loss / num_batches
        lr_scheduler.step()
        train_losses.append(avg_epoch_loss)
        print(f"\nEpoch {epoch+1}/{args.epoch} Summary:")
        print(f"  Training Loss: {avg_epoch_loss:.6f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

        # Save best model
        if avg_epoch_loss < best_train_loss:
            best_train_loss = avg_epoch_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'train_loss': avg_epoch_loss,
                'best_train_loss': best_train_loss,
                'train_losses': train_losses,
                'config': vars(args),
                'model_type': 'latent_ddim_with_res'
            }, os.path.join(args.save_dir, "best_latent_ddim_model.pth"))
            print(f"  🎉 New best model saved! Training Loss: {best_train_loss:.6f}")

        # Save periodic checkpoint
        if (epoch + 1) % 20 == 0:
            checkpoint_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'train_loss': avg_epoch_loss,
                'train_losses': train_losses,
                'best_train_loss': best_train_loss,
                'config': vars(args)
            }, checkpoint_path)
            print(f"  💾 Checkpoint saved: {checkpoint_path}")

    # Save final model
    final_model_path = os.path.join(args.save_dir, "final_latent_ddim_model.pth")
    torch.save({
        'epoch': args.epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': lr_scheduler.state_dict(),
        'train_losses': train_losses,
        'best_train_loss': best_train_loss,
        'config': vars(args),
        'model_type': 'latent_ddim_with_res'
    }, final_model_path)
    print(f"\n🎉 Latent DDIM training completed!")
    print(f"Best training loss: {best_train_loss:.6f}")
    print(f"Final model saved: {final_model_path}")

    # Test sampling
    print(f"\n🔍 Starting sampling tests...")
    best_model_path = os.path.join(args.save_dir, "best_latent_ddim_model.pth")
    if os.path.exists(best_model_path):
        best_checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        res_model.load_state_dict(torch.load(os.path.join(args.save_dir, "res_model_pretrained.pth"), map_location=device))
        
        # Test set sampling
        test_results, avg_mse = test_sampling(
            model, scheduler, test_loader, device, args, conv_encoder, 
            args.num_inference_steps, z50_mean, z50_std
        )
        test_sample_path = os.path.join(args.down_stream_dir, f"ddim_sampling_test_{int(time.time())}.pth")
        torch.save(test_results, test_sample_path)
        print(f"Test set sampling results saved: {test_sample_path}")
        
        # Training set sampling
        train_results, train_avg_mse = test_sampling(
            model, scheduler, train_loader, device, args, conv_encoder, 
            args.num_inference_steps, z50_mean, z50_std
        )
        train_sample_path = os.path.join(args.down_stream_dir, f"ddim_sampling_train_{int(time.time())}.pth")
        torch.save(train_results, train_sample_path)
        print(f"Training set sampling results saved: {train_sample_path}")
    else:
        print("Best model not found, skipping sampling tests.")
        test_results, avg_mse = None, None

    # Save training history
    history_path = os.path.join(args.down_stream_dir, "training_history.pth")
    torch.save({
        'train_losses': train_losses,
        'best_train_loss': best_train_loss,
        'final_test_mse': avg_mse,
        'config': vars(args),
        'scheduler_schedule': args.scheduler_schedule,
        'scheduler_s': args.scheduler_s
    }, history_path)
    
    print(f"\n✅ All tasks completed!")
    print(f"Configuration:")
    print(f"  Scheduler type: {args.scheduler_schedule}")
    print(f"  Scheduler s: {args.scheduler_s}")
    print(f"  Training timesteps: {args.num_train_timesteps}")
    print(f"  Inference steps: {args.num_inference_steps}")
    print(f"Model directory: {args.save_dir}")
    print(f"Results directory: {args.down_stream_dir}")
    if avg_mse is not None:
        print(f"Final test MSE: {avg_mse:.6f}")

if __name__ == "__main__":
    args = parse_args()
    main()