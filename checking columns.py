# Run this code if you notice the dataset has 42 columns instead of 43.

import pandas as pd

# Path to your dataset file
file_path = 'dataset/KDDTrain+.txt'

# Define all 43 columns for NSL-KDD (including 'label' and 'difficulty')
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
    'label',         # Attack label
    'difficulty'     # Difficulty level
]

# Load dataset with no header since file has none
df = pd.read_csv(file_path, header=None)

# Print number of columns to confirm
print(f"Number of columns in dataset: {df.shape[1]}")
print(f"Number of column names provided: {len(columns)}")

# Assign column names
df.columns = columns

# Show first 5 rows
print(df.head())