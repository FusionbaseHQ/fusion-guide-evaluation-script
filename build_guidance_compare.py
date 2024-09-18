import json
import os
import hashlib
from tqdm import tqdm
import boto3
from botocore.exceptions import ClientError

def raw_promp(text):
    start_tag = "<guidance_prompt>"
    end_tag = "</guidance_prompt>"
    
    # Find the start and end of the text between the tags
    start_index = text.find(start_tag) + len(start_tag)
    end_index = text.find(end_tag)
    
    # Extract and return the text
    if start_index != -1 and end_index != -1:
        return text[start_index:end_index].strip()
    else:
        return None  # Return None if tags are not properly found


def get_prompt_response(prompt, model_id="meta.llama3-1-8b-instruct-v1:0", guidance=None):
    # Create the cache directory if it doesn't exist
    cache_dir = 'cache'
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create a unique cache filename using SHA3 hash of prompt and model_id
    cache_key = hashlib.sha3_256(f"{prompt}{model_id}".encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")

    # Check if the response is cached
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as file:
            cached_response = json.load(file)
            return cached_response.get('response_text')
    
    # Create an Amazon Bedrock Runtime client.
    brt = boto3.client("bedrock-runtime",
        region_name="us-west-2", # Currently, the only supported region for all llama models seems like 'us-west-2'
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))

    # Start a conversation with the user message.
    if guidance:
        guidance_starter = ("Using the guidance below, focus on the primary objective: to create a perfect and readable "
                            "response to the given task. Ensure that each section is tailored specifically to the task at hand, "
                            "not to the guidance provided. The following instructions are solely intended to help achieve the main goal. "
                            "Think about each step to make sure the keypoints are covered and the goal is achieved:")
        user_message = f"{prompt} {guidance_starter}\n{guidance}"
    else:
        user_message = prompt        

    conversation = [
        {
            "role": "user",
            "content": [{"text": user_message}],
        }
    ]

    try:
        # Send the message to the model, using a basic inference configuration.
        response = brt.converse(
            modelId=model_id,
            messages=conversation,
            inferenceConfig={"maxTokens": 2048, "temperature": 0, "topP": 0.95},
        )

        # Extract the response text.
        response_text = response["output"]["message"]["content"][0]["text"]

        # Cache the response to a file
        with open(cache_file, 'w') as file:
            json.dump({'response_text': response_text}, file)

        return response_text

    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        return None
    
    
if __name__ == '__main__':
    # Those are the prompts and guidance samples
    # The guidance steps are generated with fusion-guide-12b-0.1 model (https://huggingface.co/fusionbase/fusion-guide-12b-0.1)
    # All guidance is generated with temperature=0
    with open("guidance_prompts_fusion-guide-12b-0.1_test_small.json", "r", encoding="utf-8") as f:
        prompt_samples = json.load(f)
    
    
    model_results =[]
    for prompt_sample in tqdm(prompt_samples):
        prompt = raw_promp(prompt_sample["prompt"]).strip()
        guidance = prompt_sample["guidance"]        
        
        no_guidance_405b = get_prompt_response(prompt, model_id="meta.llama3-1-405b-instruct-v1:0", guidance=None)
        no_guidance_70b = get_prompt_response(prompt, model_id="meta.llama3-1-70b-instruct-v1:0", guidance=None)
        guidance_8b = get_prompt_response(prompt, model_id="meta.llama3-1-8b-instruct-v1:0", guidance=guidance)
        
        model_results.append({"no_guidance_405b": no_guidance_405b, "no_guidance_70b": no_guidance_70b, "guidance_8b": guidance_8b, "prompt": prompt, "guidance": guidance})
        

    with open("model_results.json", "w", encoding="utf-8") as f:
        json.dump(model_results, f, indent=4, ensure_ascii=False)