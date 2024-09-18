import json
import random
import sys


def load_prompt_results(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def get_eval_prompt_template():
    with open("eval_prompt_template.txt", "r", encoding="utf-8") as f:
        return f.read()
    
if __name__ == '__main__':
    try:
        prompt_results = load_prompt_results("model_results.json")
    except FileNotFoundError:
        print("Please run the model evaluation script first.")
        sys.exit()
        
    try:
        eval_prompt_template = get_eval_prompt_template()
    except FileNotFoundError:
        print("Please provide a valid eval prompt template.")
        sys.exit()
        
        
    # Set the seed for reproducibility
    RND_SEED = 42424242   # 42, 1337, 2024, 5555, 54321, 42424242
    random.seed(RND_SEED)

    eval_prompt_structured = []
    for prompt_result in prompt_results:        
        keys = ["no_guidance_405b", "no_guidance_70b", "guidance_8b"]        
        
        # Process the text, replacing each occurrence of ||ANSWER|| with a random string and tracking positions
        positions = ['A', 'B', 'C']
        replacements = random.sample(keys, len(positions))  # Random selection without repetition
        output_text = eval_prompt_template

        eval_prompt_mappings = {}
        for i, replacement in enumerate(replacements):
            # Replace one occurrence of ||ANSWER||
            output_text = output_text.replace("||ANSWER||", prompt_result[replacement], 1)
            output_text = output_text.replace("||PROMPT||", prompt_result["prompt"], 1)
            # Storing position
            print(f"Position {positions[i]} is replaced with: {replacement}")
            
            eval_prompt_mappings[positions[i]] = replacement
        
        eval_prompt_structured.append({"prompt": output_text, **eval_prompt_mappings})
            

    with open(f"eval_prompt_structured__gpt4o-judge-rnd_seed_{RND_SEED}.json", "w", encoding="utf-8") as f:
        json.dump(eval_prompt_structured, f, indent=4, ensure_ascii=False)
        
        
        

