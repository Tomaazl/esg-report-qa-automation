#!/usr/bin/env python3
"""
Parse Any Document - Ultra Simple Version
Just change the file path below and run!
"""

from simple_document_parser import SimpleDocumentParser

# ============================================================================
# CHANGE THIS PATH TO YOUR DOCUMENT
# ============================================================================
DOCUMENT_PATH = "your_document.pdf"  # <-- PUT YOUR FILE PATH HERE
# ============================================================================

def main():
    """Parse the document specified in DOCUMENT_PATH"""
    
    # Create parser
    parser = SimpleDocumentParser()
    
    # Parse the document
    print(f"Parsing: {DOCUMENT_PATH}")
    success = parser.parse_document_to_json(DOCUMENT_PATH)
    
    if success:
        print("✅ Done! Check for generated JSON files.")
    else:
        print("❌ Failed to parse document.")

if __name__ == "__main__":
    main()
