import json
import fitz
import re
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text.strip()

def segment_text(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [sentence for sentence in sentences if sentence]

def generate_question(segment, model, tokenizer):
    input_text = f"generate questions from the following text: {segment}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return question

def extract_answer(question, context):
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2",
                           tokenizer="deepset/roberta-base-squad2")
    result = qa_pipeline(question=question, context=context)
    return result['answer']

def main(pdf_path, output_json_path):
    model_name = "valhalla/t5-small-qg-hl"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t5_model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    t5_tokenizer = T5Tokenizer.from_pretrained(model_name)

    combined_text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(combined_text)
    segments = segment_text(cleaned_text)

    qa_pairs = []
    for segment in segments:
        question = generate_question(segment, t5_model, t5_tokenizer)
        answer = extract_answer(question, segment)
        qa_pairs.append({'question': question, 'answer': answer})

    with open(output_json_path, 'w') as json_file:
        json.dump(qa_pairs, json_file)

if __name__ == "__main__":
    pdf_path = "path_to_your_pdf.pdf"  # Update this path as needed
    output_json_path = "output/qa_pairs.json"
    main(pdf_path, output_json_path)