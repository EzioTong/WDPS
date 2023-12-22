from transformers import pipeline, GPT2Tokenizer, GPT2Model
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import torch
import transformers

# model_name = "meta-llama/Llama-2-7b-chat-hf"
# model_name = "deepset/roberta-base-squad2"
# model_name = "GPT2ForQuestionAnswering"
# model_name = "rsvp-ai/bertserini-bert-base-squad"
# model_name = "llama2"
model_name = "gpt2"
# model_name = transformers.GPT2ForQuestionAnswering(
#  'mosaicml/mpt-7b',
#  trust_remote_code=True
# )

# tokenizer = AutoTokenizer.from_pretrained(model)

def invoke_language_model(question):
    generator = pipeline('text-generation', model= model_name)
    # generator = pipeline('question-answering', model = model_name, tokenizer=model_name)
    raw_text = generator(question, max_length=150)[0]['generated_text']
    # raw_text = generator([question, 'context'])
    return raw_text

# model = AutoModelForQuestionAnswering.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

def process_raw_text(raw_text):
    processed_text = raw_text.strip()
    return processed_text

def get_raw_text(question):
    return invoke_language_model(question)
def get_processed_text(raw_text):
    return process_raw_text(raw_text)

if __name__ == "__main__":
    # question = "Is Amsterdam the capital of the Netherlands?"
    # question = "Is Managua the capital of Nicaragua?"
    user_question = input("Enter the question please: ")

    raw_text_result = get_raw_text(user_question)

    processed_text_result = get_processed_text(raw_text_result)

    print("Input Question (Text A):", user_question)
    print("Raw Text Output (Text B):", raw_text_result)
    print("Processed Text:", processed_text_result)
