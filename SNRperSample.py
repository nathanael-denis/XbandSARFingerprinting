import numpy as np
import matplotlib.pyplot as plt
import os
import glob

'''
Compute the SNR for a specified IQ file

You may have to run this script multiple times for different files, as it it consuming to load IQ files in the RAM.

'''
# Parameters (adjust these based on your setup)
directory_path = 'RawIQ'  # Path to subdirectory containing IQ files
file_extension = '*'  # File extension for IQ files (e.g., *.bin)
sample_rate = 10e6  # Sample rate in Hz (e.g., 10 MHz, adjust as needed)
noise_percentile = 10  # Percentile for noise floor estimation (e.g., 10th percentile)
magnitude_threshold = None  # Set to your threshold for filtered data, or None for unfiltered

# Step 1: Get list of IQ files in the subdirectory
file_list = glob.glob(os.path.join(directory_path, file_extension))
if not file_list:
    raise FileNotFoundError(f"No files with extension {file_extension} found in {directory_path}")

# Step 2: Initialize array to store all SNR values
all_snr_db = []

# Step 3: Process each IQ file
for file_path in file_list:
    print(f"Processing file: {file_path}")
    
    # Load IQ data (assuming complex64 format)
    iq_samples = np.fromfile(file_path, dtype=np.complex64)
    if len(iq_samples) == 0:
        print(f"Warning: {file_path} is empty, skipping")
        continue
    
    # Compute magnitudes
    magnitudes = np.abs(iq_samples)
    
    # Estimate noise floor
    noise_magnitude = np.percentile(magnitudes, noise_percentile)
    noise_power = noise_magnitude**2
    
    # Compute SNR
    signal_power = magnitudes**2
    snr_db = 10 * np.log10(signal_power / noise_power)
    
    # Apply magnitude-based filtering (optional)
    if magnitude_threshold is not None:
        mask = magnitudes >= magnitude_threshold
        snr_db = snr_db[mask]
        print(f"Filtered {file_path} to {len(snr_db)} samples (from {len(magnitudes)})")
    
    # Append SNR values to the aggregate list
    all_snr_db.extend(snr_db)

# Convert to numpy array for efficiency
all_snr_db = np.array(all_snr_db)

# Step 4: Plot combined SNR distribution
plt.figure(figsize=(8, 6))
plt.hist(all_snr_db, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
plt.xlabel('SNR (dB)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('SNR Distribution of X-band SAR Signals (9.65 GHz, All Files)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Save the figure as PNG
output_file = 'snr_distribution_all_unfiltered.png' if magnitude_threshold is None else 'snr_distribution_all_filtered.png'
plt.savefig(output_file, format='png', dpi=300)
plt.close()

# Step 5: Print basic statistics
print(f"Number of files processed: {len(file_list)}")
print(f"Total samples: {len(all_snr_db)}")
print(f"Mean SNR: {np.mean(all_snr_db):.2f} dB")
print(f"Median SNR: {np.median(all_snr_db):.2f} dB")
print(f"Standard Deviation of SNR: {np.std(all_snr_db):.2f} dB")