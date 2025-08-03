import argparse
from omegaconf import OmegaConf
import numpy as np
import torch
import random
from dataset_SEED_DEAP_new import load_SEED_data, load_DEAP_data_mean_std, DEAP, SEED_condition
from torch.utils.data import DataLoader, TensorDataset
from generative.networks.nets import AutoencoderKL, PatchDiscriminator
import os
import time

def set_seed(seed: int):
    """Set seed for all RNGs"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_all_data(cfg):
    """Load all data (train+test) and return separately"""
    train_eeg, train_labels, train_mask, test_eeg, test_labels, test_mask, subject_train, subject_test = load_SEED_data(
        cfg.patient_id,
        cfg.signal_length,
        cfg.shuffle,
        cfg.dataset_dir,
    )
    print(f"Train data shape: {train_eeg.shape}")
    print(f"Test data shape: {test_eeg.shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Test labels shape: {test_labels.shape}")
    return (
        torch.from_numpy(train_eeg).float(), torch.from_numpy(train_labels),
        torch.from_numpy(test_eeg).float(), torch.from_numpy(test_labels)
    )

def mask_eeg_channels(eeg, mask_ratio=0.5, mask_channel_names=None, ch_names=None):
    """Mask EEG channels by name or randomly"""
    N, C, L = eeg.shape
    masked_eeg = eeg.clone()
    if mask_channel_names is not None and ch_names is not None:
        mask_channel_indices = []
        for name in mask_channel_names:
            if name not in ch_names:
                raise ValueError(f"Channel {name} not found in ch_names!")
            mask_channel_indices.append(ch_names.index(name))
        print(f"Masking fixed channels: {mask_channel_indices} ({mask_channel_names})")
        masked_eeg[:, mask_channel_indices, :] = 0
        num_mask = int(C * mask_ratio)
        n_fixed = len(mask_channel_indices)
        if n_fixed < num_mask:
            for i in range(N):
                remain_indices = list(set(range(C)) - set(mask_channel_indices))
                n_to_fill = num_mask - n_fixed
                if n_to_fill > 0:
                    fill_indices = np.random.choice(remain_indices, n_to_fill, replace=False)
                    masked_eeg[i, fill_indices, :] = 0
    else:
        num_mask = int(C * mask_ratio)
        print(f"Masking {num_mask}/{C} channels ({mask_ratio*100}%)")
        for i in range(N):
            idx = torch.randperm(C)[:num_mask]
            masked_eeg[i, idx, :] = 0
    return masked_eeg

def extract_z(model, data, device, batch_size=64):
    """Extract latent variable z"""
    model.eval()
    all_z = []
    with torch.no_grad():
        for i in range(0, data.shape[0], batch_size):
            batch = data[i:i+batch_size].to(device)
            _, z_mu, z_sigma = model(batch)
            all_z.append(z_mu.cpu())
            if (i // batch_size + 1) % 10 == 0:
                print(f"Processed batch {i // batch_size + 1}/{(data.shape[0] + batch_size - 1) // batch_size}")
    all_z = torch.cat(all_z, dim=0)
    print(f"Extracted z shape: {all_z.shape}")
    return all_z

def parse_args():
    parser = argparse.ArgumentParser(description="Autoencoder Z Extraction")
    parser.add_argument('--dataset_name', type=str, default='SEED', help='Dataset name')
    parser.add_argument('--patient_id', type=str, default='all', help='Patient ID')
    parser.add_argument('--shuffle', type=bool, default=False, help='Shuffle dataset')
    parser.add_argument('--num_channel', type=int, default=62, help='Number of channels')
    parser.add_argument('--signal_length', type=int, default=800, help='Signal length')
    parser.add_argument('--dataset_dir', type=str, default="", help='Dataset directory')
    parser.add_argument('--save_dir', type=str, default="", help='Model save directory')
    parser.add_argument('--down_stream_dir', type=str, default="")
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--train_batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--config_file', type=str, default="")
    parser.add_argument('--ae_num_channels', type=int, nargs='+', default=[64, 32], 
                       help='AutoencoderKL channel configuration')
    parser.add_argument('--ae_latent_channels', type=int, default=32, 
                       help='Latent channels in AutoencoderKL')
    parser.add_argument('--mask_channel_names', type=str, nargs='*', 
                        default=["AF3", "AF4", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "TP7", "CP5", "CP3", "CP1", 
                                 "CPZ", "CP2", "CP4", "CP6", "TP8", "POO7", "POO8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8"],
                        help='Channels to mask')
    return parser.parse_args()

def load_best_model(cfg, config, device):
    """Load best trained model"""
    best_model_path = cfg.save_dir
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model not found: {best_model_path}")
    print(f"Loading model from {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location=device)
    autoencoder_args = config.autoencoderkl.params
    autoencoder_args['num_channels'] = cfg.ae_num_channels
    autoencoder_args['latent_channels'] = cfg.ae_latent_channels
    model = AutoencoderKL(**autoencoder_args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    best_val_loss = checkpoint.get('best_loss', 'N/A')
    print(f"Model loaded successfully! (Val Loss: {best_val_loss})")
    print(f"Model config - channels: {cfg.ae_num_channels}, latent: {cfg.ae_latent_channels}")
    return model

def main():
    ch_names = [
        'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
        'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ',
        'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7',
        'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6',
        'PO8', 'POO7',  'O1', 'OZ', 'O2','POO8'
    ]
    args = parse_args()
    config = OmegaConf.load(args.config_file)
    cfg = OmegaConf.create()
    cfg.dataset_name = args.dataset_name
    cfg.patient_id = args.patient_id
    cfg.shuffle = args.shuffle
    cfg.num_channel = args.num_channel
    cfg.signal_length = args.signal_length
    cfg.dataset_dir = args.dataset_dir
    cfg.save_dir = args.save_dir
    cfg.down_stream_dir = args.down_stream_dir
    cfg.device = args.device
    cfg.seed = args.seed
    cfg.train_batch_size = args.train_batch_size
    cfg.config_file = args.config_file
    cfg.ae_num_channels = args.ae_num_channels
    cfg.ae_latent_channels = args.ae_latent_channels
    cfg.mask_channel_names = args.mask_channel_names
    
    if cfg.seed is not None:
        set_seed(cfg.seed)
        
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(cfg.down_stream_dir, exist_ok=True)
    
    cfg_args_path = os.path.join(cfg.down_stream_dir, "config_full.txt")
    with open(cfg_args_path, "w") as f:
        f.write("="*80 + "\n")
        f.write("Autoencoder Z Extraction Configuration\n")
        f.write("="*80 + "\n\n")
        f.write("Command Line Arguments:\n")
        f.write("-"*50 + "\n")
        parser_args = vars(args)
        for key, value in parser_args.items():
            f.write(f"  {key:<25}: {value}\n")
        f.write("\nFinal Configuration:\n")
        f.write("-"*50 + "\n")
        for key, value in cfg.items():
            f.write(f"  {key:<25}: {value}\n")
        import datetime
        f.write(f"\nConfiguration saved at: {datetime.datetime.now()}\n")
        f.write("="*80 + "\n")
    print(f"Saved config to: {cfg_args_path}")
    
    model = load_best_model(cfg, config, device)
    print("\nLoading train/test data...")
    train_eeg, train_labels, test_eeg, test_labels = get_all_data(cfg)
    
    # Process training data
    print("\n" + "="*60)
    print("Processing TRAIN data")
    print("Extracting train z100 (no masking)...")
    train_z100 = extract_z(model, train_eeg, device, cfg.train_batch_size)
    train_z100_path = os.path.join(cfg.down_stream_dir, "train_z100.pth")
    torch.save(train_z100, train_z100_path)
    print(f"Saved train_z100 to: {train_z100_path}")
    
    print("Extracting train z50 (with masking)...")
    train_masked_eeg = mask_eeg_channels(
        train_eeg, mask_ratio=0.5, mask_channel_names=cfg.mask_channel_names, ch_names=ch_names
    )
    train_z50 = extract_z(model, train_masked_eeg, device, cfg.train_batch_size)
    train_z50_path = os.path.join(cfg.down_stream_dir, "train_z50.pth")
    torch.save(train_z50, train_z50_path)
    print(f"Saved train_z50 to: {train_z50_path}")
    
    train_labels_path = os.path.join(cfg.down_stream_dir, "train_labels.pth")
    torch.save(train_labels, train_labels_path)
    print(f"Saved train labels to: {train_labels_path}")
    
    # Process test data
    print("\n" + "="*60)
    print("Processing TEST data")
    print("Extracting test z100 (no masking)...")
    test_z100 = extract_z(model, test_eeg, device, cfg.train_batch_size)
    test_z100_path = os.path.join(cfg.down_stream_dir, "test_z100.pth")
    torch.save(test_z100, test_z100_path)
    print(f"Saved test_z100 to: {test_z100_path}")
    
    print("Extracting test z50 (with masking)...")
    test_masked_eeg = mask_eeg_channels(
        test_eeg, mask_ratio=0.5, mask_channel_names=cfg.mask_channel_names, ch_names=ch_names
    )
    test_z50 = extract_z(model, test_masked_eeg, device, cfg.train_batch_size)
    test_z50_path = os.path.join(cfg.down_stream_dir, "test_z50.pth")
    torch.save(test_z50, test_z50_path)
    print(f"Saved test_z50 to: {test_z50_path}")
    
    test_labels_path = os.path.join(cfg.down_stream_dir, "test_labels.pth")
    torch.save(test_labels, test_labels_path)
    print(f"Saved test labels to: {test_labels_path}")
    
    # Save original data
    print("\n" + "="*60)
    print("Saving original and masked data")
    train_orig_path = os.path.join(cfg.down_stream_dir, "train_original.pth")
    torch.save(train_eeg, train_orig_path)
    test_orig_path = os.path.join(cfg.down_stream_dir, "test_original.pth")
    torch.save(test_eeg, test_orig_path)
    train_masked_path = os.path.join(cfg.down_stream_dir, "train_masked.pth")
    torch.save(train_masked_eeg, train_masked_path)
    test_masked_path = os.path.join(cfg.down_stream_dir, "test_masked.pth")
    torch.save(test_masked_eeg, test_masked_path)
    print(f"Saved original/masked data to: {cfg.down_stream_dir}")
    
    # Save config summary
    cfg_path = os.path.join(cfg.down_stream_dir, "config_summary.txt")
    with open(cfg_path, "w") as f:
        f.write("Extraction Configuration Summary:\n")
        f.write("="*50 + "\n")
        for k, v in cfg.items():
            f.write(f"{k}: {v}\n")
        f.write("\nData Shapes:\n")
        f.write(f"Train data: {train_eeg.shape}\n")
        f.write(f"Test data: {test_eeg.shape}\n")
        f.write(f"Train z100: {train_z100.shape}\n")
        f.write(f"Train z50: {train_z50.shape}\n")
        f.write(f"Test z100: {test_z100.shape}\n")
        f.write(f"Test z50: {test_z50.shape}\n")
    print(f"Saved config summary to: {cfg_path}")
    
    print("\n" + "="*60)
    print("All tasks completed successfully!")
    print(f"Model Config: channels={cfg.ae_num_channels}, latent={cfg.ae_latent_channels}")
    print(f"Data Shapes: Train={train_eeg.shape}, Test={test_eeg.shape}")
    print(f"Output Directory: {cfg.down_stream_dir}")

if __name__ == "__main__":
    main()