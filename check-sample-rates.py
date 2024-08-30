import os
from obspy import read_inventory
from collections import defaultdict

def check_sample_rates(folder_path):
    # Dictionary to store results
    results = defaultdict(list)
    
    # Iterate through files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.xml') or filename.endswith('.yaml'):  # Adjust file extensions as needed
            file_path = os.path.join(folder_path, filename)
            
            try:
                # Read the inventory
                inv = read_inventory(file_path)
                
                # Iterate through networks, stations, and channels
                for network in inv:
                    for station in network:
                        for channel in station:
                            sample_rate = channel.sample_rate
                            channel_id = f"{network.code}.{station.code}.{channel.location_code}.{channel.code}"
                            
                            if sample_rate == 0:
                                results['zero_sample_rate'].append((channel_id, file_path))
                            else:
                                results['non_zero_sample_rate'].append((channel_id, file_path, sample_rate))
            
            except Exception as e:
                results['errors'].append((file_path, str(e)))
    
    return results

def print_results(results):
    print("Channels with 0 sample rate:")
    for channel, file_path in results['zero_sample_rate']:
        print(f"  - {channel} (File: {file_path})")
    
    print("\nChannels with non-zero sample rate:")
    for channel, file_path, sample_rate in results['non_zero_sample_rate']:
        print(f"  - {channel}: {sample_rate} Hz (File: {file_path})")
    
    print("\nErrors encountered:")
    for file_path, error in results['errors']:
        print(f"  - File: {file_path}, Error: {error}")

# Main execution
if __name__ == "__main__":
    folder_path = input("Enter the path to the folder containing inventory files: ")
    results = check_sample_rates(folder_path)
    print_results(results)
