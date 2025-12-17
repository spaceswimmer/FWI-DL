import os
import numpy as np
import segyio
import time

def read_sgy_selective(filepath, endian='big'):
    with segyio.open(filepath, mode='r', endian=endian, ignore_geometry=True) as src:
        traces = segyio.tools.collect(src.trace[:])
        # traces = 123
        # Select only needed fields (still vectorized)
        il = src.attributes(segyio.TraceField.INLINE_3D)[:]
        xl = src.attributes(segyio.TraceField.CROSSLINE_3D)[:]
    return traces, il, xl

def find_viable_arrays(folder_path):
    """
    Cycle through .npy files in a folder and return arrays with more than 1 unique value.
    
    Args:
        folder_path (str): Path to the folder containing .npy files
    
    Returns:
        list: List of tuples containing (filename, array) for arrays with >1 unique value
    """
    valid_arrays = []
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist")
        return valid_arrays
    
    # Get all .npy files in the folder
    npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    
    if not npy_files:
        print(f"No .npy files found in '{folder_path}'")
        return valid_arrays
    
    print(f"Found {len(npy_files)} .npy files")
    
    for filename in npy_files:
        file_path = os.path.join(folder_path, filename)
        
        try:
            # Load the array
            array = np.load(file_path)
            
            # Count unique values
            unique_values = np.unique(array)
            
            # Check if array has more than 1 unique value
            if len(unique_values) > 1:
                valid_arrays.append((filename))
                print(f"✓ {filename}: {len(unique_values)} unique values")
            else:
                print(f"✗ {filename}: Only 1 unique value ({unique_values[0]})")
                
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    print(f"\nFound {len(valid_arrays)} arrays with multiple unique values")
    return valid_arrays