import json
from rich import print
import matplotlib.pyplot as plt
import numpy as np

RND_SEED = 42424242 # 42, 1337, 2024, 5555, 54321, 42424242

def count_ranks_with_combined(data):
    # Initialize counters for each model's ranks
    counts = {
        'no_guidance_70b': {'first': 0, 'second': 0, 'third': 0},
        'no_guidance_405b': {'first': 0, 'second': 0, 'third': 0},
        'guidance_8b': {'first': 0, 'second': 0, 'third': 0},
        'no_guidance_70b_or_405b': {'first': 0, 'second': 0, 'third': 0}  # Combined count
    }

    # Iterate through each dictionary in the list
    for entry in data:
        # Create a mapping for 'A', 'B', 'C' to the model names
        model_map = {
            'A': entry['A'],
            'B': entry['B'],
            'C': entry['C']
        }

        # Get ranks from gpt_judgements
        ranks = entry['gpt_judgements']['ranks']

        # Update counts for each rank
        for i, rank in enumerate(['first', 'second', 'third']):
            model_name = model_map[ranks[i]]
            counts[model_name][rank] += 1

            # Check if the model is either 'no_guidance_70b' or 'no_guidance_405b'
            if model_name in ['no_guidance_70b', 'no_guidance_405b']:
                counts['no_guidance_70b_or_405b'][rank] += 1

    return counts

if __name__ == '__main__':
    for seed in [42, 1337, 2024, 5555, 54321, 42424242]:
        RND_SEED = seed   
    
        # Load the JSON data
        with open(f"eval_prompt_structured_reasoned-gpt-4o-judge-rnd-seed-{RND_SEED}.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        # Call the function with the loaded data
        rank_counts = count_ranks_with_combined(data)
        print(rank_counts)

        # Extract the models and subcategories
        models = list(rank_counts.keys())
        subcategories = list(next(iter(rank_counts.values())).keys())

        # Define the positions of the bars
        x = np.arange(len(models))  # the label locations
        width = 0.2  # the width of the bars

        # Create the plot with a wider figure size
        fig, ax = plt.subplots(figsize=(12, 6))  # Adjust the figsize to make the chart wider

        # Define colors and opacities for the ranks
        colors = {
            'first': '#4A89FF',  # Updated color for first rank
            'second': '#FF4A4A',  # Normal color for second rank
            'third': '#95A5A6'   # Normal color for third rank
        }
        opacities = {
            'first': 1.0,  # Full opacity for first rank
            'second': 0.3,  # Semi-transparent for second rank
            'third': 0.2   # More transparent for third rank
        }

        # Plotting each subcategory's bars
        for i, subcategory in enumerate(subcategories):
            subcategory_values = [rank_counts[model][subcategory] for model in models]
            ax.bar(x + i * width, subcategory_values, width, label=subcategory,
                color=colors[subcategory], alpha=opacities[subcategory])

        # Add labels, title, and custom x-axis tick labels
        ax.set_xlabel('Models')
        ax.set_ylabel('Number of Rankings')
        ax.set_title('GPT-4o Judgement Ranks')
        ax.set_xticks(x + width)
        ax.set_xticklabels(models)
        ax.legend()

        # Save the plot to a file
        plt.savefig(f'gpt-4o-judgement-ranks-rnd-seed-{RND_SEED}.png')

        # Show the plot (optional)
        plt.show()
