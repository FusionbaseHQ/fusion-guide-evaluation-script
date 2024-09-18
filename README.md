# fusion-guide Evaluation script
This repository contains the scripts to compare the performance of different language models with and without guidance prompts. The script reads prompt samples from a JSON file, invokes different models, and saves the results to another JSON file.

Read the full article including a link to download the fusion-guide here: 

## Files
- `build_guidance_compare.py`: The main script to run the model comparison.
- `create_eval_prompts.py`: This script creates the evaluation prompts for the judge.
- `gpt-judgement.py`: This script calls GPT4o as a judge.
- `count_ranks.py`: This scripts counts and evaluates the ranking of the GPT judgement.
- `guidance_prompts_fusion-guide-12b-0.1_test_small.json`: JSON file containing the prompt samples and guidance prompts.
- `model_results.json`: Output file where the results of the model comparisons are saved.

## Requirements
- Python 3.x
- Required Python packages (can be installed via `requirements.txt` if available)

## Usage
1. Ensure you have Python 3.x installed on your machine.
2. Install the required Python packages:
3. Run the script:

## Script Details
The script performs the following steps:

1. Reads prompt samples from `guidance_prompts_fusion-guide-12b-0.1_test_small.json`.
2. For each prompt sample, it invokes three different models:
    - `meta.llama3-1-405b-instruct-v1:0` without guidance
    - `meta.llama3-1-70b-instruct-v1:0` without guidance
    - `meta.llama3-1-8b-instruct-v1:0` with guidance
3. Collects the responses from the models and appends them to a list.
4. Saves the collected results to `model_results.json`.

## Example Output
The output file `model_results.json` will contain a list of dictionaries with the following structure:

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.