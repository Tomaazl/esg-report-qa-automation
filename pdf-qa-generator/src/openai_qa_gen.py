from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal
from enum import Enum
from openai import AzureOpenAI
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION')

class QAPair(BaseModel):
    question: str
    answer: str

class QAPairList(BaseModel):
    qa_pairs: List[QAPair]

# Use QAPairList as the output_class
output_class = QAPairList


client = AzureOpenAI(api_key=API_KEY, azure_endpoint=ENDPOINT, api_version=API_VERSION)

system_prompt = "You are ESG expert and your task is to extract relevant Q&A pairs that customers might ask in RFI ESG questionaires"
def call_openai_structured(client, output_class, prompt, system_prompt=system_prompt, deployment="", temperature=0.3):
    try:
        response = client.beta.chat.completions.parse(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=deployment,
            response_format=output_class,
            temperature=temperature,
            max_tokens=16384
        )
        return response.choices[0].message.parsed
    except Exception as e:
        print(e)
        return e