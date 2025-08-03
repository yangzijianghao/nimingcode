import torch
import os
import numpy as np
from omegaconf import OmegaConf
from generative.networks.nets import AutoencoderKL

def load_data(sample_path, zcond_path, original_path):
    sample_data = torch.load(sample_path)
    zcond = torch.load(zcond_path)
    original = torch.load(original_path)
    assert original is not None and zcond is not None
    return zcond, original

def mse(x, y):
    return np.mean((x - y) ** 2)

def nmse(x, y):
    return np.sum((x - y) ** 2) / np.sum(y ** 2)

def pcc(x, y):
    x_flat = x.flatten()
    y_flat = y.flatten()
    return np.corrcoef(x_flat, y_flat)[0, 1]

def snr(x, y):
    signal_power = np.mean(x ** 2)
    noise_power = np.mean((x - y) ** 2)
    return 10 * np.log10(signal_power / (noise_power + 1e-8))

def evaluate(zcond, original, decoder, device):
    zcond = zcond.to(device)
    original = original.to(device)
    with torch.no_grad():
        recon = decoder(zcond)
    # Verify recon and original have matching shapes
    assert recon.shape == original.shape, f"Recon shape {recon.shape} != original shape {original.shape}"
    recon_np = recon.cpu().numpy()
    original_np = original.cpu().numpy()
    
    print("recon_np first 10:", recon_np[0][0][:10])
    print("original_np first 10:", original_np[0][0][:10])
    mse_score = mse(recon_np, original_np)
    nmse_score = nmse(recon_np, original_np)
    pcc_score = pcc(recon_np, original_np)
    snr_score = snr(original_np, recon_np)
    return mse_score, nmse_score, pcc_score, snr_score

def main():
    # Path configurations (all set to empty)
    train_sample_path = ""
    train_zcond_path = ""
    test_sample_path = ""
    test_zcond_path = ""
    train_original_path = ""
    test_original_path = ""
    model_path = ""
    config_file = ""

    # Load data
    train_zcond, train_original = load_data(train_sample_path, train_zcond_path, train_original_path)
    test_zcond, test_original = load_data(test_sample_path, test_zcond_path, test_original_path)

    # Load VAE decoder (inference only)
    config = OmegaConf.load(config_file)
    autoencoder_args = config.autoencoderkl.params
    autoencoder_args['num_channels'] = [64,32]
    autoencoder_args['latent_channels'] = 32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoencoderKL(**autoencoder_args).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    decoder = model.decode
    model.eval()
    for param in model.parameters():
        param.requires_grad = False  # Freeze decoder parameters

    # Evaluate test set
    print("\nTesting Set Evaluation:")
    test_mse, test_nmse, test_pcc, test_snr = evaluate(test_zcond, test_original, decoder, device)
    print(f"MSE  : {test_mse:.6f}")
    print(f"NMSE : {test_nmse:.6f}")
    print(f"PCC  : {test_pcc:.6f}")
    print(f"SNR  : {test_snr:.6f} dB")

if __name__ == "__main__":
    main()