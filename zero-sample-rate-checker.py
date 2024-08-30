import os
import csv
from obspy import read_inventory

def check_sample_rates(inventory_folder):
    results = []

    # Iterate through all files in the folder
    for filename in os.listdir(inventory_folder):
        if filename.endswith(('.xml', '.XML')):  # Assuming inventory files are XML
            file_path = os.path.join(inventory_folder, filename)
            
            # Read the inventory file
            inv = read_inventory(file_path)

            # Check each channel in the inventory
            for network in inv:
                for station in network:
                    for channel in station:
                        channel_id = f"{network.code}.{station.code}.{channel.location_code}.{channel.code}"
                        sample_rate = channel.sample_rate
                        
                        results.append({
                            'file': filename,
                            'channel': channel_id,
                            'sample_rate': sample_rate,
                            'status': 'Zero' if sample_rate == 0 else 'Non-zero'
                        })

    return results

def save_to_csv(results, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['file', 'channel', 'sample_rate', 'status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in results:
            writer.writerow(row)

def main():
    inventory_folder = "./inventory"  # Replace with your folder path
    output_file = "sample_rate_report.csv"

    results = check_sample_rates(inventory_folder)

    # Print report to console
    print("Sample Rate Report:")
    zero_count = sum(1 for r in results if r['status'] == 'Zero')
    print(f"Total channels checked: {len(results)}")
    print(f"Channels with zero sample rate: {zero_count}")
    print(f"Channels with non-zero sample rate: {len(results) - zero_count}")
    
    print("\nChannels with zero sample rate:")
    for result in results:
        if result['status'] == 'Zero':
            print(f"File: {result['file']}, Channel: {result['channel']}")

    # Save results to CSV
    save_to_csv(results, output_file)
    print(f"\nDetailed results saved to {output_file}")

if __name__ == "__main__":
    main()
