# multithreadGPT
Use GPT4 and GPT3.5 on inputs of unlimited size. Uses multithreading to process multiple chunks in parallel. Useful for tasks like Named Entity Recognition, information extraction on large books, datasets, etc.

Use cases:
- Translating a large body of text
- Extracting geographic entities from a book on the history of wars
- Summarizing a long article, textbook, or other file bit by bit.

It is designed to handle large files that may exceed OpenAI's token limits if processed as a whole. The script splits the input file into manageable pieces and sends each chunk to the OpenAI API separately at the same time. The responses are then collected and saved into an output file.

If the OpenAI rate limit is reached, the code uses exponential backoff with jitter to keep retrying until success. It is by default set to give up after three failures.

![image](https://cloud-ojq43hax6-hack-club-bot.vercel.app/0screen_shot_2023-06-11_at_8.44.36_pm.png)
## Installation

### Prerequisites

- Python 3.6 or above
- OpenAI API key (either set using echo or hard-code into the main.py script)
- Basic understanding of the command-line interface (Terminal for macOS and Linux, CMD or PowerShell for Windows)

### Steps

1. Clone the GitHub repository to your local machine. (OR might just be easier to download the main.py file and use it directly)
```bash
git clone https://github.com/your_username/openai-text-processor.git
```
2. Change directory to the cloned repository.
```bash
cd openai-text-processor
```
3. Install the required packages.

```bash
openai
tiktoken
tqdm
```

4. The script requires an OpenAI API key, which should be set as an environment variable. You can do this in bash by running the following command:

```bash
export OPENAI_KEY=your_openai_key
```
Replace `your_openai_key` with your actual OpenAI API key.

**Note:** The way to set environment variables can vary depending on your operating system and shell. Please consult the appropriate documentation if the above method does not apply to your situation.

## Usage

### Command-Line Interface

You can use the OpenAI Text Processor through the command-line interface. The usage is as follows:

```bash
python main.py -i INPUT_FILE -o OUTPUT_FILE -l LOG_FILE -m MODEL -c CHUNKSIZE -t TOKENS -v TEMPERATURE -p PROMPT
```

Where:

- `INPUT_FILE` is the path to the input file. This argument is required.
- `OUTPUT_FILE` is the path to the output file. This argument is required.
- `LOG_FILE` is the path to the log file. This argument is required.
- `MODEL` is the OpenAI model to use (default is 'gpt-3.5-turbo-0301'). Alternative: gpt-4-0314. Better quality but slower and more expensive. 
- `CHUNKSIZE` is the maximum number of tokens per chunk (default is 1000). This shouldn't be too large (>4000) or OpenAI will be overloaded. A safe size is under 3000 tokens. Your prompt length also counts for the OpenAI token limit.
- `TOKENS` is the maximum tokens per API call (default is 100). shorter will be faster. but could terminate too early.
- `TEMPERATURE` is the variability (temperature) for OpenAI model (default is 0.0). 0.0 is probably best if you are going for highest accuracy
- `PROMPT` is the prompt for the OpenAI model. This argument is required. Counts towards the 4k token limit for OpenAI API calls.

### Example

```bash
python main.py -i input.txt -o output.txt -l log.txt -m 'gpt-3.5-turbo' -c 500 -t 200 -v 0.5 -p 'Translate English to French:'
```

This will process the file `input.txt`, using the model 'gpt-3.5-turbo', a chunk size of 500 tokens, a maximum of 200 tokens per API call, a temperature of 0.5, and the prompt 'Translate English to French:'. The results will be saved in `output.txt` and the logs in `log.txt`.

## License

[MIT](LICENSE.md)
```

Inspired by https://github.com/emmethalm/infiniteGPT from Emmet Halm.
