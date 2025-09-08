import fitz
import pytesseract
from PIL import Image
import io
import re
import json
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, RobertaTokenizer, RobertaForSequenceClassification

# Load the pre-trained models and tokenizers
t5_model_name = "valhalla/t5-small-qg-hl"
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)

classification_model_name = "roberta-base"
classification_tokenizer = RobertaTokenizer.from_pretrained(classification_model_name)
classification_model = RobertaForSequenceClassification.from_pretrained(classification_model_name)

def extract_text_and_images(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
        images = page.get_images(full=True)
        for img in images:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            text += pytesseract.image_to_string(image)
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text.strip()

def segment_sentences(text):
    return re.split(r'(?<=[.!?]) +', text)

def generate_question(segment):
    input_text = f"generate questions from the following text: {segment}"
    input_ids = t5_tokenizer.encode(input_text, return_tensors='pt')
    outputs = t5_model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
    question = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return question

def classify_difficulty(question):
    inputs = classification_tokenizer(question, return_tensors='pt')
    outputs = classification_model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    difficulty = ["Easy", "Medium", "Hard"][predicted_class]
    return difficulty

def process_pdf(pdf_path):
    combined_text = extract_text_and_images(pdf_path)
    cleaned_text = clean_text(combined_text)
    segments = segment_sentences(cleaned_text)

    qa_pairs = []
    for segment in segments:
        question = generate_question(segment)
        difficulty = classify_difficulty(question)
        qa_pairs.append({'question': question, 'difficulty': difficulty})

    return qa_pairs

def save_to_json(qa_pairs, output_path):
    with open(output_path, 'w') as json_file:
        json.dump(qa_pairs, json_file)