import os
import re
import scipy.io as sio
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import train_test_split


def load_SEED_data(pat_id, signal_length=800, shuffle=False, filepath=None):
    """
    Load and process SEED dataset
    
    :param pat_id: Patient ID, "all" for all patients, "1_" for specific patient
    :param signal_length: Signal length (default 800)
    :param shuffle: Whether to shuffle data
    :param filepath: Directory containing .mat files
    :return: (X_train, y_train, train_mask, X_test, y_test, test_mask)
    """
    # Set default data directory if filepath is None
    if filepath is None:
        filepath = "./data"  # Default path
    
    # Define trial labels (1=positive, 0=neutral, -1=negative)
    trial_labels = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
    false_data = []  # List of files to skip
    
    # Get all .mat files in directory
    all_files = [f for f in os.listdir(filepath) if f.endswith('.mat')]
    
    # Filter files based on patient ID
    if pat_id == "all":
        mat_files = all_files
    else:
        if not pat_id.endswith('_'):
            pat_id += '_'
        mat_files = [f for f in all_files if f.startswith(pat_id)]
    
    mat_files.sort()  # Ensure consistent order

    # Initialize lists to store samples and labels
    all_samples = []
    all_labels = []

    # Process each .mat file
    for file_name in mat_files:
        file_base = os.path.splitext(file_name)[0]
        if file_base in false_data:
            print(f"Skipping abnormal file: {file_name}")
            continue
        
        file_path = os.path.join(filepath, file_name)
        mat_data = sio.loadmat(file_path)

        # Find EEG data keys (eeg1 to eeg15)
        eeg_keys = [key for key in mat_data.keys() if re.match(r'.*_eeg[1-9][0-9]?$', key)]

        # Process each EEG trial
        for key in eeg_keys:
            eeg_data = mat_data[key]  # Current EEG data (62, T)
            T = eeg_data.shape[1]  # Current segment length
            num_samples = T // signal_length  # Number of full samples
            
            # Extract samples of specified length
            for i in range(num_samples):
                sample = eeg_data[:, i * signal_length:(i + 1) * signal_length]
                all_samples.append(sample)
                
                # Get corresponding label from trial number
                trial_index = int(re.search(r'_eeg(\d+)', key).group(1)) - 1
                all_labels.append(trial_labels[trial_index])

        print(f"Processed file: {file_name}, Total samples: {len(all_samples)}")

    # Check if any data was found
    if len(all_samples) == 0:
        raise ValueError(f"No data found for patient ID '{pat_id}'")

    # Convert to NumPy arrays
    all_samples = np.array(all_samples)  # (N, 62, signal_length)
    all_labels = np.array(all_labels)   # (N,)

    # Split data (80% train, 20% test)
    if pat_id == "all":
        X_train, X_test, y_train, y_test = train_test_split(
            all_samples, all_labels, train_size=0.8, random_state=42, shuffle=shuffle
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            all_samples, all_labels, train_size=0.8, random_state=42, shuffle=shuffle
        )

    # Create all-ones masks (all data points valid)
    train_mask = np.ones_like(X_train)  # (N_train, 62, signal_length)
    test_mask = np.ones_like(X_test)    # (N_test, 62, signal_length)

    return (
        X_train,          # Train data (N_train, 62, signal_length)
        y_train,          # Train labels (N_train,)
        train_mask,       # Train mask (all 1s)
        X_test,           # Test data (N_test, 62, signal_length)
        y_test,           # Test labels (N_test,)
        test_mask,        # Test mask (all 1s)
    )
    
    
class SEED_condition(Dataset):
    """
    SEED dataset class with support for conditional and masked training
    """

    def __init__(
        self,
        signal_length=1001,
        data_array=None,
        label_array=None,
        mask_array=None,
        train_mean=None,
        train_std=None,
        conditional_training=True,
        lr_channels=None,
        hr_channels=None,
        masked_training=True,
    ):
        super().__init__()
        self.conditional_training = conditional_training
        self.masked_training = masked_training

        _num_time_windows, self.num_channels, self.signal_length = data_array.shape
        assert self.signal_length == signal_length
        self.data_array = data_array
        self.label_array = label_array
        self.mask_array = mask_array  # Values to ignore (outliers)
        self.train_mean = train_mean
        self.train_std = train_std
        self.hr_channels = hr_channels
        self.lr_channels = lr_channels

    def __getitem__(self, index):
        """Get a sample with optional conditioning and masking"""
        return_dict = {}
        return_dict["signal"] = torch.from_numpy(np.float32(self.data_array[index]))  # Signal without mask
        return_dict["label"] = torch.tensor([np.float32(self.label_array[index])])
        cond = self.get_cond(index=index)  # Get conditioning information
        
        if cond is not None and self.masked_training:
            return_dict["mask"] = torch.from_numpy(np.float32(self.mask_array[index]))
            # Apply mask to conditioning
            cond[self.num_channels :] *= return_dict["mask"]
            cond[: self.num_channels] *= return_dict["mask"]
            return_dict["cond"] = cond
            
        if cond is not None:
            return_dict["cond"] = cond
            
        if self.masked_training:
            return_dict["mask"] = torch.from_numpy(np.float32(self.mask_array[index]))
            
        return return_dict

    def get_cond(self, index=0):
        """Create conditioning information"""
        cond = None
        if self.conditional_training:
            condition_mask = torch.zeros(self.num_channels, self.signal_length)
            # Randomly select channels for conditioning
            num_cond_channels = np.random.randint(
                int(self.num_channels * 0.3), int(self.num_channels * 0.7)
            )
            cond_channel = list(
                np.random.choice(
                    self.num_channels,
                    size=num_cond_channels,
                    replace=False,
                )
            )
            # Mark selected channels with 1.0
            condition_mask[cond_channel, :] = 1.0
            condition_signal = (
                torch.from_numpy(np.float32(self.data_array[index])) * condition_mask
            )
            cond = torch.cat((condition_mask, condition_signal), dim=0)
        return cond  # First half indicates available channels, second half is masked signal

    def get_condition_fixed(self, index=0):
        """Create fixed conditioning information"""
        cond = None
        if self.conditional_training:
            condition_mask = torch.zeros(self.num_channels, self.signal_length)
            # Fixed conditioning channels (low-resolution channels)
            cond_channel = [self.hr_channels.index(ch) for ch in self.lr_channels if ch in self.hr_channels]
            condition_mask[cond_channel, :] = 1.0  # Mark selected channels
            condition_signal = (
                torch.from_numpy(np.float32(self.data_array[index])) * condition_mask
            )
            cond = torch.cat((condition_mask, condition_signal), dim=0)
        return cond

    def get_train_mean_and_std(self):
        """Get normalization parameters"""
        return torch.from_numpy(np.float32(self.train_mean)), torch.from_numpy(
            np.float32(self.train_std)
        )

    def __len__(self):
        return len(self.data_array)
    
    
class DEAP(Dataset):
    """
    DEAP dataset class with support for conditional and masked training
    """

    def __init__(
        self,
        signal_length=512,
        masked_training=True,
        data_array=None,
        label_array=None,
        mask_array=None,
        train_mean=None,
        train_std=None,
    ):
        super().__init__()
        self.masked_training = masked_training

        _num_time_windows, self.num_channels, self.signal_length = data_array.shape
        assert self.signal_length == signal_length
        self.data_array = data_array
        self.label_array = label_array
        self.mask_array = mask_array  # Values to ignore (outliers)
        self.train_mean = train_mean
        self.train_std = train_std

    def __getitem__(self, index):
        """Get a sample with optional conditioning and masking"""
        return_dict = {}
        return_dict["signal"] = torch.from_numpy(np.float32(self.data_array[index]))
        return_dict["label"] = torch.tensor([np.float32(self.label_array[index])])
        cond = self.get_cond(index=index)
        
        if cond is not None and self.masked_training:
            return_dict["mask"] = torch.from_numpy(np.float32(self.mask_array[index]))
            # Apply mask to conditioning
            cond[self.num_channels :] *= return_dict["mask"]
            cond[: self.num_channels] *= return_dict["mask"]
            return_dict["cond"] = cond
            
        if cond is not None:
            return_dict["cond"] = cond
            
        return_dict["mask"] = torch.from_numpy(np.float32(self.mask_array[index]))
        return return_dict

    def get_cond(self, index=0):
        """Create conditioning information"""
        cond = None
        condition_mask = torch.zeros(self.num_channels, self.signal_length)
        # Randomly select channels for conditioning
        num_cond_channels = np.random.randint(self.num_channels + 1)
        cond_channel = list(
            np.random.choice(
                self.num_channels,
                size=num_cond_channels,
                replace=False,
            )
        )
        # Mark selected channels with 1.0
        condition_mask[cond_channel, :] = 1.0
        condition_signal = (
            torch.from_numpy(np.float32(self.data_array[index])) * condition_mask
        )
        cond = torch.cat((condition_mask, condition_signal), dim=0)
        return cond

    def get_train_mean_and_std(self):
        """Get normalization parameters"""
        return torch.from_numpy(np.float32(self.train_mean)), torch.from_numpy(
            np.float32(self.train_std)
        )

    def __len__(self):
        return len(self.data_array)
    
    
def load_DEAP_data_mean_std(pat_id, window_size=512, root_dir=""):
    """
    Load and process DEAP dataset
    
    :param pat_id: Patient ID, "all" for all patients
    :param window_size: Window size for segmentation
    :param root_dir: Root directory containing data files
    :return: (train_eeg, train_labels, train_mask, test_eeg, test_labels, test_mask, channel_means, channel_stds)
    """
    baseline_samples = 384  # Baseline period to remove
    train_ratio = 0.8  # Train-test split ratio
    
    eeg_data = []
    labels = []
    subject_ids = []

    # Determine subject list to process
    if pat_id == "all":
        subj_list = [f"s{i:02d}" for i in range(1, 33)]
    else:
        subj_list = [pat_id]

    for subj_id in subj_list:
        file_path = os.path.join(root_dir, f"{subj_id}_all_trials.npz")
        
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        
        # Load data
        data = np.load(file_path)
        subj_eeg = data['eeg']  # (40, 32, 8064)
        subj_labels = data['labels']  # (40, 4)
        
        # Process each trial
        for trial_idx in range(subj_eeg.shape[0]):
            # Remove baseline (first 3 seconds)
            trial_eeg = subj_eeg[trial_idx, :, baseline_samples:]  # (32, 7680)
            trial_label = subj_labels[trial_idx]  # (4,)
            
            # Calculate number of windows
            num_windows = trial_eeg.shape[1] // window_size
            
            # Segment into windows
            for win_idx in range(num_windows):
                start = win_idx * window_size
                end = start + window_size
                window_eeg = trial_eeg[:, start:end]  # (32, window_size)
                
                # Add to results
                eeg_data.append(window_eeg)
                # Repeat labels for each window
                labels.append(np.tile(trial_label, (32, 1)))  # (32, 4)
                subject_ids.append(subj_id)
    
    # Convert to NumPy arrays
    eeg_data = np.array(eeg_data)  # (N, 32, window_size)
    labels = np.array(labels)      # (N, 32, 4)
    
    # Split into train and test sets
    train_eeg, test_eeg, train_labels, test_labels = train_test_split(
        eeg_data, labels, train_size=train_ratio, random_state=42
    )
    
    # Calculate per-channel mean and std from training data
    channel_means = np.mean(train_eeg, axis=(0, 2))  # (32,)
    channel_stds = np.std(train_eeg, axis=(0, 2))    # (32,)
    
    # Create all-ones masks
    train_mask = np.ones_like(train_eeg)  # (N_train, 32, window_size)
    test_mask = np.ones_like(test_eeg)    # (N_test, 32, window_size)

    # Normalize data
    train_eeg = (train_eeg - channel_means[None, :, None]) / channel_stds[None, :, None]
    test_eeg = (test_eeg - channel_means[None, :, None]) / channel_stds[None, :, None]

    print("Train EEG shape:", train_eeg.shape)
    print("Train labels shape:", train_labels.shape)
    print("Train mask shape:", train_mask.shape)
    print("Test EEG shape:", test_eeg.shape)
    print("Test labels shape:", test_labels.shape)
    print("Test mask shape:", test_mask.shape)
    
    
    return (
        train_eeg,          # Train data (N_train, 32, window_size)
        train_labels,        # Train labels (N_train, 32, 4)
        train_mask,          # Train mask (all 1s)
        test_eeg,            # Test data (N_test, 32, window_size)
        test_labels,         # Test labels (N_test, 32, 4)
        test_mask,           # Test mask (all 1s)
        channel_means,       # Channel means (32,)
        channel_stds,        # Channel stds (32,)
    )