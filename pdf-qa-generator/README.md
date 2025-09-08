# PDF QA Generator

This project is designed to extract sentences from a PDF document, generate question-answer pairs from the content, and save the results in JSON format. 

## Project Structure

```
pdf-qa-generator
├── src
│   ├── main.py          # Entry point of the application
│   └── utils
│       └── pdf_processing.py  # Utility functions for PDF processing
├── requirements.txt     # List of dependencies
├── README.md            # Documentation for the project
└── output
    └── qa_pairs.json    # Output file for question-answer pairs
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd pdf-qa-generator
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Place your PDF document in the appropriate directory.
2. Run the application:
   ```
   python src/main.py <path_to_your_pdf>
   ```

3. After execution, the generated question-answer pairs will be saved in `output/qa_pairs.json`.

## Contributing

Feel free to submit issues or pull requests for improvements or bug fixes. 

## License

This project is licensed under the MIT License.