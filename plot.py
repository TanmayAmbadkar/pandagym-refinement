import matplotlib.pyplot as plt
import os
import numpy as np

def find_folders(directory):
    """
    Find all folder names in the specified directory.

    Args:
    - directory (str): The directory path.

    Returns:
    - list: A list containing names of all folders in the directory.
    """
    # Initialize an empty list to store folder names
    folders = []

    # Iterate over all items (files and folders) in the directory
    for item in os.listdir(directory):
        # Check if the item is a directory
        if os.path.isdir(os.path.join(directory, item)):
            folders.append(item)

    return folders

# Example usage:
directory = "./results"
folders = find_folders(directory)

map_results = {}

for folder in folders:

    exp_name, _, episodes, seed = folder.split("-")
    episodes = int(episodes[5:])
    seed = int(seed[5:])
    
    with open(directory + '/' + folder + "/result.txt", "r") as f:
        result = f.readlines()
        
    
    if (episodes, "DIRL") not in map_results:
        map_results[(episodes, "DIRL")] = []
    if (episodes,  "Refinement") not in map_results:
        map_results[(episodes, "Refinement")] = []
    
    map_results[(episodes, "DIRL")].append(float(result[0].split(" ")[-1]))
    map_results[(episodes, "Refinement")].append(float(result[1].split(" ")[-1]))
    

# Sort dictionary by the number of episodes
sorted_data = dict(sorted(map_results.items(), key=lambda item: item[0][0]))

# Initialize colors for DIRL and Refinement
colors = {'DIRL': 'red', 'Refinement': 'blue'}

# Initialize lists to store points for connecting lines
refinement_points = {'x': [], 'y': []}
dirl_points = {'x': [], 'y': []}

# Initialize legend flags
dirl_legend_added = False
refinement_legend_added = False

# Iterate through the sorted dictionary
for key, values in sorted_data.items():
    timesteps, legend = key
    mean = np.mean(values)
    std = np.std(values)
    
    # Plot mean with error bars
    plt.errorbar(timesteps, mean, yerr=std, fmt='o', color=colors[legend], label=legend if (legend == 'DIRL' and not dirl_legend_added) or (legend == 'Refinement' and not refinement_legend_added) else '_nolegend_')
    
    # Store points for connecting lines
    if legend == 'Refinement':
        refinement_points['x'].append(timesteps)
        refinement_points['y'].append(mean)
        refinement_legend_added = True
    else:
        dirl_points['x'].append(timesteps)
        dirl_points['y'].append(mean)
        dirl_legend_added = True

print(sorted_data)

# Plot connecting lines for DIRL
plt.plot(dirl_points['x'], dirl_points['y'], color='red', linestyle='dotted')

# Plot connecting lines for Refinement
plt.plot(refinement_points['x'], refinement_points['y'], color='blue', linestyle='dotted')

# Add labels and title
# plt.xlabel('Timesteps')
# plt.ylabel('Probability')
# plt.title('Mean Probability with Error Bars')
plt.legend( loc='upper left')
plt.grid(True)
plt.show()