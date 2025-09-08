# Document Question Parser

A Python tool that uses Docling to parse questions from various document formats (PDF, DOCX, PPTX, XLSX) and formats them into a JSON structure similar to `test_questions.json`.

## Features

- **Multi-format Support**: Parses PDF, DOCX, PPTX, and XLSX files
- **Intelligent Question Extraction**: Uses multiple regex patterns to identify questions
- **Text Cleaning**: Removes document artifacts and normalizes formatting
- **Duplicate Detection**: Automatically removes duplicate questions
- **Flexible Output**: Saves in the same format as `test_questions.json` or with additional metadata

## Installation

1. Install Docling (required for document parsing):
```bash
pip install docling
```

2. Install additional dependencies if needed:
```bash
pip install pathlib dataclasses
```

## Usage

### Basic Usage

```python
from document_question_parser import DocumentQuestionParser

# Initialize the parser
parser = DocumentQuestionParser()

# Parse a single document
questions = parser.parse_document("path/to/your/document.pdf")

# Parse multiple documents
documents = [
    "document1.pdf",
    "survey.docx", 
    "presentation.pptx",
    "spreadsheet.xlsx"
]
all_questions = parser.parse_documents(documents)

# Save in test_questions.json format
parser.save_to_json(all_questions, "extracted_questions.json")
```

### Extract from Text

```python
# Extract questions from plain text (useful for testing)
sample_text = """
1. What are your climate targets?
2. How do you measure emissions?
3. Does your company have diversity policies?
"""

questions = parser.extract_questions_from_text(sample_text)
parser.save_to_json(questions, "text_questions.json")
```

### Run Example

```bash
python example_usage.py
```

This will:
1. Extract questions from sample ESG text
2. Look for documents in the current directory
3. Show example code for processing specific documents

## Output Format

### Standard Format (same as test_questions.json)
```json
{
  "test_questions": [
    {
      "id": 1,
      "question": "What are your company's climate targets?"
    },
    {
      "id": 2,
      "question": "How do you measure carbon emissions?"
    }
  ]
}
```

### Detailed Format (with metadata)
```json
{
  "extracted_questions": [
    {
      "id": 1,
      "question": "What are your company's climate targets?",
      "source_file": "document.pdf",
      "confidence": 1.0
    }
  ],
  "metadata": {
    "total_questions": 1,
    "extraction_method": "docling + regex patterns"
  }
}
```

## Question Detection Patterns

The parser uses multiple regex patterns to identify questions:

1. **Direct questions**: Text ending with `?`
2. **Question words**: Starting with What, How, When, Where, Why, Who, Which, Does, etc.
3. **Numbered questions**: `1. What is...?`
4. **Quoted questions**: `"How does...?"`
5. **Questions after colons**: `Question: What is...?`

## Text Cleaning Features

- Removes document artifacts (page numbers, headers)
- Normalizes whitespace and line breaks
- Removes section headers attached to questions
- Removes numbering and bullet points
- Ensures proper capitalization
- Adds question marks if missing

## Question Validation

Questions are validated to ensure quality:
- Minimum length requirement (10 characters)
- Must end with question mark
- Should contain question words or be properly structured
- Duplicate detection and removal

## File Structure

- `document_question_parser.py` - Main parser class
- `example_usage.py` - Usage examples and testing
- `README_document_parser.md` - This documentation

## Supported Document Types

- **PDF**: Text-based PDFs (scanned PDFs may need OCR preprocessing)
- **DOCX**: Microsoft Word documents
- **PPTX**: PowerPoint presentations
- **XLSX**: Excel spreadsheets (text content from cells)

## Error Handling

The parser includes robust error handling:
- Gracefully handles missing files
- Continues processing if one document fails
- Provides informative error messages
- Falls back to text extraction if Docling is unavailable

## Customization

You can customize the parser by:

1. **Adding new question patterns**:
```python
parser.question_patterns.append(r'(Custom pattern here)')
```

2. **Modifying validation rules**:
```python
def custom_validation(self, question):
    # Your custom validation logic
    return True
```

3. **Custom text cleaning**:
```python
def custom_clean_text(self, text):
    # Your custom cleaning logic
    return cleaned_text
```

## Troubleshooting

### Docling Installation Issues
If you encounter issues installing Docling:
```bash
# Try upgrading pip first
pip install --upgrade pip

# Install Docling
pip install docling

# If still failing, try with specific Python version
python -m pip install docling
```

### No Questions Found
If no questions are extracted:
1. Check if the document contains actual questions with `?`
2. Verify the document is text-based (not scanned images)
3. Try the text extraction method first to test patterns
4. Check the console output for processing errors

### Memory Issues with Large Documents
For very large documents:
1. Process documents one at a time
2. Use text extraction on specific sections
3. Consider splitting large documents

## Examples

See `example_usage.py` for complete working examples including:
- Text-based question extraction
- Document discovery and processing
- Error handling
- Output formatting

## Contributing

To improve the parser:
1. Add new question patterns for better detection
2. Improve text cleaning for specific document types
3. Add support for additional file formats
4. Enhance validation rules

## License

This tool is provided as-is for document processing and question extraction tasks.
