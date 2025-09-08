import json
import re
import json 
from openai_qa_gen import *
import re
import PyPDF2

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text.strip()


def main(pdf_path, output_json_path):

    cleaned_text = clean_text(extract_text_from_pdf(pdf_path))
    # Example prompt
    prompt = f"Extract a relevant ESG question and answer from the provided report: {cleaned_text}"

    # Call the function
    result = call_openai_structured(
        client=client,
        output_class=output_class,
        prompt=prompt,
        system_prompt=system_prompt,
        deployment="esg-qa",  # Replace with your actual deployment name
        temperature=0.3
    )
    # Convert result to dict
    result_dict = result.dict()

    with open(output_json_path, 'w') as json_file:
        json.dump(result_dict, json_file)




if __name__ == "__main__":
    pdf_path = "Sustainability_Highlights_2024.pdf"  # Update this path as needed
    output_json_path = "pdf-qa-generator/output/qa_pairs.json"
    main(pdf_path, output_json_path)
    print(f"QA pairs saved to {output_json_path}")