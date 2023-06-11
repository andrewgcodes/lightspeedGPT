import os
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
import tiktoken
import argparse
import logging
from tqdm import tqdm
import time
import random

logging.basicConfig(level=logging.INFO) # Setup the logging system

# You can also just paste your key in as a string.
openai.api_key = os.getenv('OPENAI_KEY')

def load_text(file_path):
    try:
        with open(file_path, 'r') as file:  # Open the file in read mode
            return file.read()  # Read and return file content
    except Exception as e:  # If an error occurs while reading the file
        logging.error(f"Failed to load file {file_path}: {str(e)}")
        raise  # Re-raise the exception to be handled by the caller

# Function to initialize output and log files (Create or ERASE existing files)
def initialize_files(output_file, log_file):
    try:
        open(output_file, 'w').close()  # Create or clear the output file
        open(log_file, 'w').close()     # Create or clear the log file
    except Exception as e:
        logging.error(f"Failed to initialize files {output_file}, {log_file}: {str(e)}")
        raise  # Re-raise the exception to be handled by the caller

# Function to save response to a file
def save_to_file(responses, output_file):
    try:
        with open(output_file, 'w') as file:  # Open the file in write mode
            for response in responses:  # Loop through all the responses
                file.write(response + '\n')  # Write each response followed by a newline
    except Exception as e:  
        logging.error(f"Failed to save to file {output_file}: {str(e)}")
        raise  # Re-raise the exception to be handled by the caller

# Function to log messages to a file
def log_to_file(log_file, message):
    try:
        with open(log_file, 'a') as file:  # Open the log file in append mode
            file.write(message + '\n')  # Append the log message followed by a newline
    except Exception as e:
        logging.error(f"Failed to log to file {log_file}: {str(e)}")
        raise  # Re-raise the exception to be handled by the caller

# Function to call OpenAI API with rate limit handling and retries
def call_openai_api(chunk, model, max_tokens, temperature, prompt):
    for i in range(3):  # Retry the API call up to 3 times
        try:
            # YOU CAN CHANGE THE MODEL TO GPT4 or GPT3
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": chunk},
                ],
                max_tokens=max_tokens,
                n=1,
                stop=None,
                temperature=temperature,
            )
            
            return response.choices[0]['message']['content'].strip()
        
        except openai.error.RateLimitError:  # If rate limit is exceeded
            wait_time = (2 ** i) + random.random()  # Exponential backoff with jitter
            logging.warning(f"Rate limit exceeded. Retrying after {wait_time} seconds.")
            time.sleep(wait_time)  # Wait before retrying
        except Exception as e:  # If any other error occurs
            logging.error(f"API call failed: {str(e)}")
            return None  # Return None for failure
    logging.error("Failed to call OpenAI API after multiple retries due to rate limiting.")
    return None  # Return None for failure

# Function to split a text into chunks
def split_into_chunks(text, model, tokens=500):
    encoding = tiktoken.encoding_for_model(model) 
    words = encoding.encode(text)  # Encode the text into tokens
    chunks = []  # Initialize the chunks list
    for i in range(0, len(words), tokens):  # Loop through the tokens in steps of 'tokens'
        # Decode each chunk of tokens into text and add it to the chunks list
        chunks.append(''.join(encoding.decode(words[i:i + tokens])))
    return chunks

def process_chunks(input_file, output_file, log_file, model, chunksize, max_tokens, temperature, prompt):
    initialize_files(output_file, log_file)  # Initialize output and log files

    text = load_text(input_file)  # Load the input text
    chunks = split_into_chunks(text, model, tokens=chunksize)  # Split the text into chunks
    nCh = len(chunks)  # Get the number of chunks
    print(str(nCh) + " chunks.")
    log_to_file(log_file, f"Number of chunks: {nCh}") 

    with ThreadPoolExecutor() as executor:  # Create a ThreadPoolExecutor
        # Submit tasks to the executor
        futures = {executor.submit(call_openai_api, chunk, model, max_tokens, temperature, prompt): chunk for chunk in chunks}
        responses = []  # Initialize the responses list
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
            response = future.result()  # Get the result from the future
            if response is None:  # If the API call failed
                log_to_file(log_file, f"Failed to process chunk {future}")
            else:
                responses.append(response)  # Add the response to the responses list
                log_to_file(log_file, f"Successfully processed chunk {future}")
                    
    save_to_file(responses, output_file)  # Save the responses to a file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process text using OpenAI API.')
    # Parse command line arguments
    parser.add_argument('-i', '--input', required=True, help='Input file path') 
    parser.add_argument('-o', '--output', required=True, help='Output file path')
    parser.add_argument('-l', '--log', required=True, help='Log file path')
    parser.add_argument('-m', '--model', default='gpt-3.5-turbo-0301', help='OpenAI model to use') # Can also use gpt-4-0314
    parser.add_argument('-c', '--chunksize', type=int, default=500, help='Maximum tokens per chunk') # This shouldn't be too large (>4000) or OpenAI will be overloaded. A safe size is under 3000 tokens. Your prompt length also counts for the OpenAI token limit.
    parser.add_argument('-t', '--tokens', type=int, default=200, help='Maximum tokens per API call') # shorter will be faster. but could terminate too early.
    parser.add_argument('-v', '--temperature', type=float, default=0.5, help='Variability (temperature) for OpenAI model') # 0.0 is probably best if you are going for highest accuracy
    parser.add_argument('-p', '--prompt', required=True, help='Prompt') # Instructions for GPT. This counts into the 4k token limit.

    args = parser.parse_args()  # Parse the command line arguments

    # Call the main function with the parsed arguments
    process_chunks(args.input, args.output, args.log, args.model, args.chunksize, args.tokens, args.temperature, args.prompt)

