import json
from rich import print
import os
import hashlib
from openai import OpenAI
from tqdm import tqdm
client = OpenAI()


RND_SEED = 42424242  # 42, 1337, 2024, 5555, 54321, 42424242

with open(f"eval_prompt_structured__gpt4o-judge-rnd_seed_{RND_SEED}.json", "r", encoding="utf-8") as f:
    eval_prompts = json.load(f)


def chat_with_cache(client, eval_prompts, seed):
    # Ensure cache directory exists
    cache_dir = "cache/evaluation/"
    os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_filename(prompt, seed):
        # Create a unique hash based on the seed and the prompt
        hash_input = (seed + prompt).encode('utf-8')
        prompt_hash = hashlib.sha256(hash_input).hexdigest()
        # Define the cache file path
        return os.path.join(cache_dir, f"{prompt_hash}.json")

    for eval_prompt in tqdm(eval_prompts):
        try:
            # Retrieve the prompt content
            prompt_text = eval_prompt["prompt"]
            cache_file = get_cache_filename(prompt_text, seed)

            # Check if the response is already cached
            if os.path.exists(cache_file):
                # Load the cached response
                with open(cache_file, 'r') as file:
                    cached_response = json.load(file)
                eval_prompt["gpt_judgements"] = cached_response
                print("Loaded from cache:", cached_response)
            else:
                # Call the API if not cached
                response = client.chat.completions.create(
                    model="gpt-4o", # o1-preview
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt_text
                                }
                            ]
                        }
                    ],
                    temperature=0,
                    max_tokens=4096,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    response_format={
                        "type": "json_object"
                    }
                )                
                response_data = response.choices[0].message.content.strip().replace("```json", "").replace("```", "")
                
                # Parse and store the response
                response_data = json.loads(response_data)
                eval_prompt["gpt_judgements"] = response_data

                # Save the response to the cache
                with open(cache_file, 'w') as file:
                    json.dump(response_data, file)
        except Exception as e:
            print(f"Error: {e}")
            continue

    with open(f"eval_prompt_structured_reasoned-gpt-4o-judge-rnd-seed-{RND_SEED}.json", "w", encoding="utf-8") as f:
        json.dump(eval_prompts, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    chat_with_cache(client, eval_prompts, f"gpt-4o-judge-rnd-seed-{RND_SEED}")