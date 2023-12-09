from transformers import pipeline, GPT2Tokenizer, GPT2Model
from transformers import AutoTokenizer
import torch
import transformers

model = "meta-llama/Llama-2-7b-chat-hf"

# tokenizer = AutoTokenizer.from_pretrained(model)

# Function to invoke the language model
def invoke_language_model(question):
    # Using a text-generation pipeline
    generator = pipeline('text-generation', model='llama2')
    raw_text = generator(question, max_length=200)[0]['generated_text']
    return raw_text

# Function to process raw text
def process_raw_text(raw_text):
    # Your processing steps here
    processed_text = raw_text.strip()
    return processed_text

# Example usage
if __name__ == "__main__":
    # Sample question
    # question = "Is Amsterdam the capital of the Netherlands?"
    # question = "Is Managua the capital of Nicaragua?"
    question = input("Enter the question: ")

    # Call the language model
    raw_text = invoke_language_model(question)

    # Process the raw text
    processed_text = process_raw_text(raw_text)

    # Display the results
    print("Input Question (Text A):", question)
    print("Raw Text Output (Text B):", raw_text)
    print("Processed Text:", processed_text)
