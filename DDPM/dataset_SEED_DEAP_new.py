import os
import re
import scipy.io as sio
import numpy as np
import torch
import xarray as xr
from scipy import io
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import train_test_split


def load_SEED_data(pat_id, signal_length=800, shuffle=False, filepath=None):
    """
    Load and process SEED dataset. When shuffle=True, read from preprocessed folder.
    
    :param pat_id: Patient ID, "all" for all patients, "4" for specific patient
    :param signal_length: Signal length, default 800
    :param shuffle: Whether to use shuffled data (read from preprocessed folder)
    :param filepath: Directory containing .mat files (only used when shuffle=False)
    :return: (X_train, y_train, train_mask, X_test, y_test, test_mask, subject_train, subject_test)
    """
    
    if shuffle:
        # Read from preprocessed folder
        print(f"Loading shuffled data from preprocessed folder...")
        
        # Determine data folder based on pat_id
        if pat_id == "all":
            data_folder = ""
        else:
            data_folder = f""
        
        print(f"Data folder: {data_folder}")
        
        # Check if folder exists
        if not os.path.exists(data_folder):
            raise FileNotFoundError(f"Preprocessed data folder not found: {data_folder}")
        
        try:
            # Load NumPy data
            X_train = np.load(os.path.join(data_folder, "X_train.npy"))
            y_train = np.load(os.path.join(data_folder, "y_train.npy"))
            train_mask = np.load(os.path.join(data_folder, "train_mask.npy"))
            subject_train = np.load(os.path.join(data_folder, "subject_train.npy"))
            
            X_test = np.load(os.path.join(data_folder, "X_test.npy"))
            y_test = np.load(os.path.join(data_folder, "y_test.npy"))
            test_mask = np.load(os.path.join(data_folder, "test_mask.npy"))
            subject_test = np.load(os.path.join(data_folder, "subject_test.npy"))
            
            print(f"Successfully loaded data from preprocessed folder!")
            print(f"  Train set: {X_train.shape}, Test set: {X_test.shape}")
            print(f"  Train labels: {y_train.shape}, Test labels: {y_test.shape}")
            print(f"  Train subjects: {np.unique(subject_train)}, Test subjects: {np.unique(subject_test)}")
            
            return (
                X_train,          # Train data (N_train, 62, 800)
                y_train,          # Train labels (N_train,)
                train_mask,       # Train mask (N_train, 62, 800), all 1s
                X_test,           # Test data (N_test, 62, 800)
                y_test,           # Test labels (N_test,)
                test_mask,        # Test mask (N_test, 62, 800), all 1s
                subject_train,    # Train subject IDs (N_train,)
                subject_test,     # Test subject IDs (N_test,)
            )
            
        except Exception as e:
            print(f"Failed to load data from preprocessed folder: {e}")
            print(f"Hint: Please run main() function to generate preprocessed data")
            raise
    
    else:
        # Original logic to read from .mat files
        print(f"Loading data from .mat files...")
        
        # Define default data directory if filepath is None
        if filepath is None:
            filepath = "./data"  # Default path
        
        # Define trial labels (1=positive, 0=neutral, -1=negative)
        trial_labels = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
        false_data = []
        
        # Read filtered trials file
        filtered_file = ''
        valid_trials = {}  # {filename: [trial_names]}
        
        with open(filtered_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    fname, trial = parts
                    if fname not in valid_trials:
                        valid_trials[fname] = []
                    valid_trials[fname].append(trial)
        
        # Initialize lists to store all samples and corresponding labels/subjects
        all_samples = []
        all_labels = []
        all_subjects = []  # Store subject ID for each sample

        # Iterate through valid files and trials
        for file_name, trial_list in valid_trials.items():
            # Check if in false_data
            file_base = os.path.splitext(file_name)[0]
            if file_base in false_data:
                print(f"Skipping abnormal file: {file_name}")
                continue
            
            # Filter by pat_id
            if pat_id != "all":
                if not pat_id.endswith('_'):
                    pat_id += '_'
                if not file_name.startswith(pat_id):
                    continue
            
            # Extract subject ID (numeric part from filename)
            subject_id = re.match(r'(\d+)_.*', file_name).group(1)
            
            file_path = os.path.join(filepath, file_name)
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue
                
            mat_data = sio.loadmat(file_path)
            
            # Iterate through valid trials in this file
            for trial_name in trial_list:
                if trial_name not in mat_data:
                    print(f"Trial {trial_name} not found in file {file_name}")
                    continue
                    
                eeg_data = mat_data[trial_name]  # Current EEG data (62, T)
                T = eeg_data.shape[1]  # Current segment length
                
                # Skip if data length < signal_length
                if T < signal_length:
                    print(f"Warning: {file_name} trial {trial_name} length {T} < {signal_length}, skipping")
                    continue
                
                # Calculate number of windows
                num_windows = T // signal_length
                
                # Calculate center-aligned start position if remainder
                if T % signal_length != 0:
                    start_offset = (T % signal_length) // 2
                else:
                    start_offset = 0
                
                # Window processing
                for i in range(num_windows):
                    start_idx = start_offset + i * signal_length
                    end_idx = start_idx + signal_length
                    sample = eeg_data[:, start_idx:end_idx]  # Extract sample of signal_length
                    
                    all_samples.append(sample)
                    
                    # Get corresponding label from trial number
                    trial_index = int(re.search(r'_eeg(\d+)', trial_name).group(1)) - 1
                    all_labels.append(trial_labels[trial_index])
                    
                    # Record subject ID
                    all_subjects.append(subject_id)
            
            print(f"Processed file: {file_name}, Trials processed: {len(trial_list)}, Total samples: {len(all_samples)}")

        # Check if any data found
        if len(all_samples) == 0:
            raise ValueError(f"No valid data found for patient ID '{pat_id}'")

        # Convert to NumPy arrays
        all_samples = np.array(all_samples)  # (N, 62, 800)
        all_labels = np.array(all_labels)   # (N,)
        all_subjects = np.array(all_subjects)  # (N,) subject ID array

        print(f"Total samples loaded: {len(all_samples)}")
        print(f"Sample shape: {all_samples.shape}")
        print(f"Included subject IDs: {np.unique(all_subjects)}")

        if pat_id == "all":    # Data splitting
            # Split by subject ID
            test_ids = ['13', '14', '15']
            train_mask = np.isin(all_subjects, test_ids, invert=True)
            test_mask = np.isin(all_subjects, test_ids)
            X_train = all_samples[train_mask]
            y_train = all_labels[train_mask]
            subject_train = all_subjects[train_mask]
            X_test = all_samples[test_mask]
            y_test = all_labels[test_mask]
            subject_test = all_subjects[test_mask]
        else:
            # For single patient ID, direct split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test, subject_train, subject_test = train_test_split(
                all_samples, all_labels, all_subjects, train_size=0.8, random_state=42, shuffle=False
            )

        # Create all-ones masks (all data points valid)
        train_mask = np.ones_like(X_train)  # (N_train, 62, 800)
        test_mask = np.ones_like(X_test)    # (N_test, 62, 800)

        print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"Train subject distribution: {np.unique(subject_train, return_counts=True)}")
        print(f"Test subject distribution: {np.unique(subject_test, return_counts=True)}")

        return (
            X_train,          # Train data (N_train, 62, 800)
            y_train,          # Train labels (N_train,)
            train_mask,       # Train mask (N_train, 62, 800), all 1s
            X_test,           # Test data (N_test, 62, 800)
            y_test,           # Test labels (N_test,)
            test_mask,        # Test mask (N_test, 62, 800), all 1s
            subject_train,    # Train subject IDs (N_train,)
            subject_test,     # Test subject IDs (N_test,)
        )
        
        
        
def load_SEED_data_6815(pat_id, signal_length=800, shuffle=False, filepath=None):
    """
    Load and process SEED dataset. When shuffle=True, read from preprocessed folder.
    
    :param pat_id: Patient ID, "all" for all patients, "4" for specific patient
    :param signal_length: Signal length, default 800
    :param shuffle: Whether to use shuffled data (read from preprocessed folder)
    :param filepath: Directory containing .mat files (only used when shuffle=False)
    :return: (X_train, y_train, train_mask, X_test, y_test, test_mask, subject_train, subject_test)
    """
    
    if shuffle:
        # Read from preprocessed folder
        print(f"Loading shuffled data from preprocessed folder...")
        
        # Determine data folder based on pat_id
        if pat_id == "all":
            data_folder = ""
        else:
            data_folder = f""
        
        print(f"Data folder: {data_folder}")
        
        # Check if folder exists
        if not os.path.exists(data_folder):
            raise FileNotFoundError(f"Preprocessed data folder not found: {data_folder}")
        
        try:
            # Load NumPy data
            X_train = np.load(os.path.join(data_folder, "X_train.npy"))
            y_train = np.load(os.path.join(data_folder, "y_train.npy"))
            train_mask = np.load(os.path.join(data_folder, "train_mask.npy"))
            subject_train = np.load(os.path.join(data_folder, "subject_train.npy"))
            
            X_test = np.load(os.path.join(data_folder, "X_test.npy"))
            y_test = np.load(os.path.join(data_folder, "y_test.npy"))
            test_mask = np.load(os.path.join(data_folder, "test_mask.npy"))
            subject_test = np.load(os.path.join(data_folder, "subject_test.npy"))
            
            print(f"Successfully loaded data from preprocessed folder!")
            print(f"  Train set: {X_train.shape}, Test set: {X_test.shape}")
            print(f"  Train labels: {y_train.shape}, Test labels: {y_test.shape}")
            print(f"  Train subjects: {np.unique(subject_train)}, Test subjects: {np.unique(subject_test)}")
            
            return (
                X_train,          # Train data (N_train, 62, 800)
                y_train,          # Train labels (N_train,)
                train_mask,       # Train mask (N_train, 62, 800), all 1s
                X_test,           # Test data (N_test, 62, 800)
                y_test,           # Test labels (N_test,)
                test_mask,        # Test mask (N_test, 62, 800), all 1s
                subject_train,    # Train subject IDs (N_train,)
                subject_test,     # Test subject IDs (N_test,)
            )
            
        except Exception as e:
            print(f"Failed to load data from preprocessed folder: {e}")
            print(f"Hint: Please run main() function to generate preprocessed data")
            raise
    
    else:
        # Original logic to read from .mat files
        print(f"Loading data from .mat files...")
        
        # Define default data directory if filepath is None
        if filepath is None:
            filepath = "./data"  # Default path
        
        # Define trial labels (1=positive, 0=neutral, -1=negative)
        trial_labels = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
        false_data = []
        
        # Read filtered trials file
        filtered_file = ''
        valid_trials = {}  # {filename: [trial_names]}
        
        with open(filtered_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    fname, trial = parts
                    if fname not in valid_trials:
                        valid_trials[fname] = []
                    valid_trials[fname].append(trial)
        
        # Initialize lists to store all samples and corresponding labels/subjects
        all_samples = []
        all_labels = []
        all_subjects = []  # Store subject ID for each sample

        # Iterate through valid files and trials
        for file_name, trial_list in valid_trials.items():
            # Check if in false_data
            file_base = os.path.splitext(file_name)[0]
            if file_base in false_data:
                print(f"Skipping abnormal file: {file_name}")
                continue
            
            # Filter by pat_id
            if pat_id != "all":
                if not pat_id.endswith('_'):
                    pat_id += '_'
                if not file_name.startswith(pat_id):
                    continue
            
            # Extract subject ID (numeric part from filename)
            subject_id = re.match(r'(\d+)_.*', file_name).group(1)
            
            file_path = os.path.join(filepath, file_name)
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue
                
            mat_data = sio.loadmat(file_path)
            
            # Iterate through valid trials in this file
            for trial_name in trial_list:
                if trial_name not in mat_data:
                    print(f"Trial {trial_name} not found in file {file_name}")
                    continue
                    
                eeg_data = mat_data[trial_name]  # Current EEG data (62, T)
                T = eeg_data.shape[1]  # Current segment length
                
                # Skip if data length < signal_length
                if T < signal_length:
                    print(f"Warning: {file_name} trial {trial_name} length {T} < {signal_length}, skipping")
                    continue
                
                # Calculate number of windows
                num_windows = T // signal_length
                
                # Calculate center-aligned start position if remainder
                if T % signal_length != 0:
                    start_offset = (T % signal_length) // 2
                else:
                    start_offset = 0
                
                # Window processing
                for i in range(num_windows):
                    start_idx = start_offset + i * signal_length
                    end_idx = start_idx + signal_length
                    sample = eeg_data[:, start_idx:end_idx]  # Extract sample of signal_length
                    
                    all_samples.append(sample)
                    
                    # Get corresponding label from trial number
                    trial_index = int(re.search(r'_eeg(\d+)', trial_name).group(1)) - 1
                    all_labels.append(trial_labels[trial_index])
                    
                    # Record subject ID
                    all_subjects.append(subject_id)
            
            print(f"Processed file: {file_name}, Trials processed: {len(trial_list)}, Total samples: {len(all_samples)}")

        # Check if any data found
        if len(all_samples) == 0:
            raise ValueError(f"No valid data found for patient ID '{pat_id}'")

        # Convert to NumPy arrays
        all_samples = np.array(all_samples)  # (N, 62, 800)
        all_labels = np.array(all_labels)   # (N,)
        all_subjects = np.array(all_subjects)  # (N,) subject ID array

        print(f"Total samples loaded: {len(all_samples)}")
        print(f"Sample shape: {all_samples.shape}")
        print(f"Included subject IDs: {np.unique(all_subjects)}")

        if pat_id == "all":    # Data splitting
            # Split by subject ID
            test_ids = ['6', '8', '15']
            train_mask = np.isin(all_subjects, test_ids, invert=True)
            test_mask = np.isin(all_subjects, test_ids)
            X_train = all_samples[train_mask]
            y_train = all_labels[train_mask]
            subject_train = all_subjects[train_mask]
            X_test = all_samples[test_mask]
            y_test = all_labels[test_mask]
            subject_test = all_subjects[test_mask]
        else:
            # For single patient ID, direct split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test, subject_train, subject_test = train_test_split(
                all_samples, all_labels, all_subjects, train_size=0.8, random_state=42, shuffle=False
            )

        # Create all-ones masks (all data points valid)
        train_mask = np.ones_like(X_train)  # (N_train, 62, 800)
        test_mask = np.ones_like(X_test)    # (N_test, 62, 800)

        print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"Train subject distribution: {np.unique(subject_train, return_counts=True)}")
        print(f"Test subject distribution: {np.unique(subject_test, return_counts=True)}")

        return (
            X_train,          # Train data (N_train, 62, 800)
            y_train,          # Train labels (N_train,)
            train_mask,       # Train mask (N_train, 62, 800), all 1s
            X_test,           # Test data (N_test, 62, 800)
            y_test,           # Test labels (N_test,)
            test_mask,        # Test mask (N_test, 62, 800), all 1s
            subject_train,    # Train subject IDs (N_train,)
            subject_test,     # Test subject IDs (N_test,)
        )
    
    
    
    
    
class SEED_condition(Dataset):
    """
    SEED dataset class
    Provides class information and supports conditional and masked training
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
        return_dict = {}
        return_dict["signal"] = torch.from_numpy(np.float32(self.data_array[index]))  # Signal without mask
        return_dict["label"] = torch.tensor([np.float32(self.label_array[index])])
        cond = self.get_cond(index=index)  # Only need to change here
        if cond is not None and self.masked_training:
            return_dict["mask"] = torch.from_numpy(np.float32(self.mask_array[index]))
            cond[self.num_channels :] *= return_dict["mask"]
            cond[: self.num_channels] *= return_dict["mask"]  # Condition with mask applied
            # Used to eliminate outliers
            return_dict["cond"] = cond
        if cond is not None:
            return_dict["cond"] = cond
        if self.masked_training:
            return_dict["mask"] = torch.from_numpy(np.float32(self.mask_array[index]))
        return return_dict

    def get_cond(self, index=0):
        cond = None
        if self.conditional_training:
            condition_mask = torch.zeros(self.num_channels, self.signal_length)
            num_cond_channels = np.random.randint(int(self.num_channels * 0.3), int(self.num_channels*0.7))
            cond_channel = list(
                np.random.choice(
                    self.num_channels,
                    size=num_cond_channels,
                    replace=False,
                )
            )
            # Cond channels marked with 1.0
            condition_mask[cond_channel, :] = 1.0
            condition_signal = (
                torch.from_numpy(np.float32(self.data_array[index])) * condition_mask
            )
            cond = torch.cat((condition_mask, condition_signal), dim=0)
        return cond  # First half indicates available channels, second half is original data * conditional_mask
    
    def get_condition_guding(self, index=0):
        cond = None
        if self.conditional_training:
            condition_mask = torch.zeros(self.num_channels, self.signal_length)
            # Fixed cond_channel as indices corresponding to lr_channels
            cond_channel = [self.hr_channels.index(ch) for ch in self.lr_channels if ch in self.hr_channels]

            # Mark selected channels with 1.0
            condition_mask[cond_channel, :] = 1.0

            # Generate conditional_signal
            condition_signal = (
                torch.from_numpy(np.float32(self.data_array[index])) * condition_mask
            )

            # Concatenate condition_mask and condition_signal
            cond = torch.cat((condition_mask, condition_signal), dim=0)

        return cond  # First half indicates available channels, second half is original data * conditional_mask

    def get_train_mean_and_std(self):
        return torch.from_numpy(np.float32(self.train_mean)), torch.from_numpy(
            np.float32(self.train_std)
        )  # Return mean and std

    def __len__(self):
        return len(self.data_array)
    
    
    
class DEAP(Dataset):
    """
    DEAP dataset class
    Provides class information and supports conditional and masked training
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
        return_dict = {}
        return_dict["signal"] = torch.from_numpy(np.float32(self.data_array[index]))
        return_dict["label"] = torch.tensor([np.float32(self.label_array[index])])
        cond = self.get_cond(index=index)
        if cond is not None and self.masked_training:
            return_dict["mask"] = torch.from_numpy(np.float32(self.mask_array[index]))
            cond[self.num_channels :] *= return_dict["mask"]
            cond[: self.num_channels] *= return_dict["mask"]
            # Used to eliminate outliers
            return_dict["cond"] = cond
        if cond is not None:
            return_dict["cond"] = cond
        return_dict["mask"] = torch.from_numpy(np.float32(self.mask_array[index]))
        return return_dict

    def get_cond(self, index=0):
        cond = None
        condition_mask = torch.zeros(self.num_channels, self.signal_length)
        num_cond_channels = np.random.randint(self.num_channels + 1)
        cond_channel = list(
            np.random.choice(
                self.num_channels,
                size=num_cond_channels,
                replace=False,
            )
        )
        # Cond channels marked with 1.0
        condition_mask[cond_channel, :] = 1.0
        condition_signal = (
            torch.from_numpy(np.float32(self.data_array[index])) * condition_mask
        )
        cond = torch.cat((condition_mask, condition_signal), dim=0)
        return cond  # First half indicates available channels, second half is original data * conditional_mask

    def get_train_mean_and_std(self):
        return torch.from_numpy(np.float32(self.train_mean)), torch.from_numpy(
            np.float32(self.train_std)
        )  # Return mean and std

    def __len__(self):
        return len(self.data_array)
    
    
    
def load_DEAP_data_mean_std(pat_id, window_size=512, root_dir=""):
    
    baseline_samples=384
    train_ratio=0.8
    
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
            print(f"Warning: {file_path} not found. Skipping.")
            continue
        
        # Load data
        data = np.load(file_path)
        subj_eeg = data['eeg']  # (40, 32, 8064)
        subj_labels = data['labels']  # (40, 4)
        
        # Iterate through each experiment
        for trial_idx in range(subj_eeg.shape[0]):
            # Remove baseline (first 3 seconds)
            trial_eeg = subj_eeg[trial_idx, :, baseline_samples:]  # (32, 8064-384)
            trial_label = subj_labels[trial_idx]  # (4,)
            
            # Calculate available windows
            num_windows = trial_eeg.shape[1] // window_size
            
            # Window processing
            for win_idx in range(num_windows):
                start = win_idx * window_size
                end = start + window_size
                window_eeg = trial_eeg[:, start:end]  # (32, 128)
                
                # Add to results
                eeg_data.append(window_eeg)
                # Repeat labels to match window count
                labels.append(np.tile(trial_label, (32, 1)))  # (32, 4)
                subject_ids.append(subj_id)
    
    # Convert to NumPy arrays
    eeg_data = np.array(eeg_data)  # (N, 32, 128)
    labels = np.array(labels)      # (N, 32, 4)
    
    # Split into 8:2 train/test
    train_eeg, test_eeg, train_labels, test_labels = train_test_split(
        eeg_data, labels, train_size=train_ratio, random_state=42
    )
    
    # Calculate per-channel mean and std for train set
    # train_eeg: (N_train, 32, 128)
    channel_means = np.mean(train_eeg, axis=(0, 2))  # Average over samples and time, shape (32,)
    channel_stds = np.std(train_eeg, axis=(0, 2))   # Std over samples and time, shape (32,)
    
    # Create masks, all set to 1
    train_mask = np.ones_like(train_eeg)  # (N_train, 32, 128)
    test_mask = np.ones_like(test_eeg)    # (N_test, 32, 128)

    train_eeg = (train_eeg - channel_means[None, :, None]) / channel_stds[None, :, None]
    test_eeg = (test_eeg - channel_means[None, :, None]) / channel_stds[None, :, None]

    print("train_eeg shape:", train_eeg.shape)
    print("train_labels shape:", train_labels.shape)
    print("train_mask shape:", train_mask.shape)
    print("test_eeg shape:", test_eeg.shape)
    print("test_labels shape:", test_labels.shape)
    print("test_mask shape:", test_mask.shape)
    
    
    return (
        train_eeg,          # Train data (N_train, 32, 128)
        train_labels,       # Train labels (N_train, 32, 4)
        train_mask,         # Train mask (N_train, 32, 128), all 1s
        test_eeg,           # Test data (N_test, 32, 128)
        test_labels,         # Test labels (N_test, 32, 4)
        test_mask,          # Test mask (N_test, 32, 128), all 1s
        channel_means,      # Mean (32,)
        channel_stds,       # Std (32,)
    )






def main():
    """
    Main function: Load SEED data and save all results
    """
    import os
    import torch
    
    # Set parameters
    RANDOM_SEED = 42
    SHUFFLE = True
    PATIENT_ID = '4'
    SIGNAL_LENGTH = 800
    FILEPATH = ''
    SAVE_DIR = ''
    
    print("="*80)
    print("SEED Data Loading and Saving Program")
    print("="*80)
    print(f"Parameters:")
    print(f"  Random Seed: {RANDOM_SEED}")
    print(f"  Shuffle: {SHUFFLE}")
    print(f"  Patient ID: {PATIENT_ID}")
    print(f"  Signal Length: {SIGNAL_LENGTH}")
    print(f"  Data Path: {FILEPATH}")
    print(f"  Save Directory: {SAVE_DIR}")
    print("-"*80)
    
    # Set random seed
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    # Create save directory
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Created save directory: {SAVE_DIR}")
    
    # Load SEED data
    print("\nLoading SEED data...")
    try:
        (
            X_train,          # Train data (N_train, 62, 800)
            y_train,          # Train labels (N_train,)
            train_mask,       # Train mask (N_train, 62, 800), all 1s
            X_test,           # Test data (N_test, 62, 800)
            y_test,           # Test labels (N_test,)
            test_mask,        # Test mask (N_test, 62, 800), all 1s
            subject_train,    # Train subject IDs (N_train,)
            subject_test,     # Test subject IDs (N_test,)
        ) = load_SEED_data(
            pat_id=PATIENT_ID,
            signal_length=SIGNAL_LENGTH,
            shuffle=SHUFFLE,
            filepath=FILEPATH
        )
        
        print("✅ SEED data loaded successfully!")
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return
    
    # Print data statistics
    print(f"\n📊 Data statistics:")
    print(f"  Train set:")
    print(f"    Data shape: {X_train.shape}")
    print(f"    Label shape: {y_train.shape}")
    print(f"    Mask shape: {train_mask.shape}")
    print(f"    Subject ID shape: {subject_train.shape}")
    print(f"    Label distribution: {np.unique(y_train, return_counts=True)}")
    print(f"    Subject distribution: {np.unique(subject_train, return_counts=True)}")
    
    print(f"  Test set:")
    print(f"    Data shape: {X_test.shape}")
    print(f"    Label shape: {y_test.shape}")
    print(f"    Mask shape: {test_mask.shape}")
    print(f"    Subject ID shape: {subject_test.shape}")
    print(f"    Label distribution: {np.unique(y_test, return_counts=True)}")
    print(f"    Subject distribution: {np.unique(subject_test, return_counts=True)}")
    
    # Save all data
    print(f"\nSaving data to: {SAVE_DIR}")
    
    # Save train set data
    train_data_path = os.path.join(SAVE_DIR, "X_train.npy")
    np.save(train_data_path, X_train)
    print(f"  ✅ Train data saved: {train_data_path}")
    
    train_labels_path = os.path.join(SAVE_DIR, "y_train.npy")
    np.save(train_labels_path, y_train)
    print(f"  ✅ Train labels saved: {train_labels_path}")
    
    train_mask_path = os.path.join(SAVE_DIR, "train_mask.npy")
    np.save(train_mask_path, train_mask)
    print(f"  ✅ Train mask saved: {train_mask_path}")
    
    subject_train_path = os.path.join(SAVE_DIR, "subject_train.npy")
    np.save(subject_train_path, subject_train)
    print(f"  ✅ Train subject IDs saved: {subject_train_path}")
    
    # Save test set data
    test_data_path = os.path.join(SAVE_DIR, "X_test.npy")
    np.save(test_data_path, X_test)
    print(f"  ✅ Test data saved: {test_data_path}")
    
    test_labels_path = os.path.join(SAVE_DIR, "y_test.npy")
    np.save(test_labels_path, y_test)
    print(f"  ✅ Test labels saved: {test_labels_path}")
    
    test_mask_path = os.path.join(SAVE_DIR, "test_mask.npy")
    np.save(test_mask_path, test_mask)
    print(f"  ✅ Test mask saved: {test_mask_path}")
    
    subject_test_path = os.path.join(SAVE_DIR, "subject_test.npy")
    np.save(subject_test_path, subject_test)
    print(f"  ✅ Test subject IDs saved: {subject_test_path}")
    
    # Save PyTorch format data (for convenience)
    print(f"\nSaving PyTorch format data...")
    
    torch_train_data = torch.from_numpy(X_train).float()
    torch_train_labels = torch.from_numpy(y_train).long()
    torch_test_data = torch.from_numpy(X_test).float()
    torch_test_labels = torch.from_numpy(y_test).long()
    
    torch.save(torch_train_data, os.path.join(SAVE_DIR, "X_train.pth"))
    torch.save(torch_train_labels, os.path.join(SAVE_DIR, "y_train.pth"))
    torch.save(torch_test_data, os.path.join(SAVE_DIR, "X_test.pth"))
    torch.save(torch_test_labels, os.path.join(SAVE_DIR, "y_test.pth"))
    print(f"  ✅ PyTorch format data saved")
    
    # Save data configuration info
    config_info = {
        'random_seed': RANDOM_SEED,
        'shuffle': SHUFFLE,
        'patient_id': PATIENT_ID,
        'signal_length': SIGNAL_LENGTH,
        'data_filepath': FILEPATH,
        'save_directory': SAVE_DIR,
        'train_data_shape': X_train.shape,
        'test_data_shape': X_test.shape,
        'train_labels_shape': y_train.shape,
        'test_labels_shape': y_test.shape,
        'train_subjects': np.unique(subject_train).tolist(),
        'test_subjects': np.unique(subject_test).tolist(),
        'train_label_distribution': dict(zip(*np.unique(y_train, return_counts=True))),
        'test_label_distribution': dict(zip(*np.unique(y_test, return_counts=True))),
        'num_channels': X_train.shape[1],
        'signal_length_actual': X_train.shape[2],
    }
    
    # Save config info to txt file
    config_path = os.path.join(SAVE_DIR, "data_config.txt")
    with open(config_path, 'w') as f:
        f.write("SEED Dataset Configuration\n")
        f.write("="*50 + "\n\n")
        
        f.write("Loading Parameters:\n")
        f.write(f"  Random Seed: {config_info['random_seed']}\n")
        f.write(f"  Shuffle: {config_info['shuffle']}\n")
        f.write(f"  Patient ID: {config_info['patient_id']}\n")
        f.write(f"  Signal Length: {config_info['signal_length']}\n")
        f.write(f"  Data Path: {config_info['data_filepath']}\n")
        
        f.write(f"\nData Shapes:\n")
        f.write(f"  Train data: {config_info['train_data_shape']}\n")
        f.write(f"  Test data: {config_info['test_data_shape']}\n")
        f.write(f"  Train labels: {config_info['train_labels_shape']}\n")
        f.write(f"  Test labels: {config_info['test_labels_shape']}\n")
        f.write(f"  Channels: {config_info['num_channels']}\n")
        f.write(f"  Signal length: {config_info['signal_length_actual']}\n")
        
        f.write(f"\nSubject Distribution:\n")
        f.write(f"  Train subjects: {config_info['train_subjects']}\n")
        f.write(f"  Test subjects: {config_info['test_subjects']}\n")
        
        f.write(f"\nLabel Distribution:\n")
        f.write(f"  Train set: {config_info['train_label_distribution']}\n")
        f.write(f"  Test set: {config_info['test_label_distribution']}\n")
        
        import datetime
        f.write(f"\nSaved at: {datetime.datetime.now()}\n")
    
    print(f"  ✅ Configuration saved: {config_path}")
    
    # Save file list
    file_list_path = os.path.join(SAVE_DIR, "file_list.txt")
    with open(file_list_path, 'w') as f:
        f.write("Saved File List:\n")
        f.write("="*30 + "\n")
        f.write("NumPy Format:\n")
        f.write("  X_train.npy - Train data\n")
        f.write("  y_train.npy - Train labels\n")
        f.write("  train_mask.npy - Train mask\n")
        f.write("  subject_train.npy - Train subject IDs\n")
        f.write("  X_test.npy - Test data\n")
        f.write("  y_test.npy - Test labels\n")
        f.write("  test_mask.npy - Test mask\n")
        f.write("  subject_test.npy - Test subject IDs\n")
        f.write("\nPyTorch Format:\n")
        f.write("  X_train.pth - Train data\n")
        f.write("  y_train.pth - Train labels\n")
        f.write("  X_test.pth - Test data\n")
        f.write("  y_test.pth - Test labels\n")
        f.write("\nConfiguration Files:\n")
        f.write("  data_config.txt - Data configuration\n")
        f.write("  file_list.txt - File list description\n")
    
    print(f"  ✅ File list saved: {file_list_path}")
    
    print(f"\n🎉 All data saved successfully!")
    print(f"📁 Save location: {SAVE_DIR}")
    print(f"📋 Saved files:")
    print(f"  - 8 NumPy files (.npy)")
    print(f"  - 4 PyTorch files (.pth)")
    print(f"  - 2 configuration files (.txt)")
    print("="*80)


if __name__ == "__main__":
    main()