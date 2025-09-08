# 📋 ESG Q&A Document Processor - Streamlit Web App

A user-friendly web interface for processing documents, extracting questions, and matching them to answers from your knowledge base.

## 🚀 Quick Start

### Option 1: Simple Launch
```bash
python run_streamlit_app.py
```

### Option 2: Direct Launch
```bash
streamlit run streamlit_qa_app.py
```

### Option 3: Manual Installation
```bash
pip install streamlit
streamlit run streamlit_qa_app.py
```

## 📋 Features

### 📤 **Document Upload**
- Drag & drop or browse to upload files
- Supports multiple formats: PDF, Excel, Word, PowerPoint
- File validation and preview

### 🔍 **Question Extraction**
- Automatic question detection using AI (Docling + regex)
- Real-time processing progress
- Handles complex document structures

### 🎯 **Answer Matching**
- TF-IDF similarity matching
- Configurable number of top matches (1-10)
- Similarity scores for each match

### 📊 **Interactive Results**
- **Summary View**: Best matches overview in table format
- **Detailed View**: All matches with rankings in tabular format
- **Question-Answer Format**: Each question as header with answer candidates as paragraphs
- **Statistics**: Match rates and counts
- **Sortable/Filterable**: Easy data exploration

### 💾 **Export Options**
- **JSON**: Complete results with metadata
- **Excel**: Multi-sheet workbook with:
  - Summary sheet (best matches)
  - All matches sheet (detailed view)
  - Metadata sheet (processing info)

## 🖥️ **User Interface**

### **Sidebar Configuration**
- **Knowledge Base**: Set Q&A pairs file path
- **Matching Settings**: Configure top-k matches
- **Format Support**: View supported file types

### **Main Area**
- **Left Column**: File upload and processing
- **Right Column**: Results display with multiple view options and download

## 📁 **File Requirements**

### **Input Documents**
- **PDF**: `.pdf`
- **Excel**: `.xlsx`, `.xls`
- **Word**: `.docx`, `.doc`
- **PowerPoint**: `.pptx`, `.ppt`

### **Knowledge Base**
- **Q&A Pairs File**: `qa_pairs.json` (default path: `pdf-qa-generator/output/qa_pairs.json`)
- **Format**: 
  ```json
  {
    "qa_pairs": [
      {
        "question": "Your question here?",
        "answer": "Your answer here."
      }
    ]
  }
  ```

## 🔧 **How It Works**

1. **📤 Upload**: User uploads document via web interface
2. **🔍 Extract**: Questions automatically extracted using Docling + regex patterns
3. **🎯 Match**: Questions matched to Q&A knowledge base using TF-IDF similarity
4. **📊 Display**: Results shown in interactive tables with statistics
5. **💾 Export**: Download results as JSON or Excel files

## 📊 **Output Formats**

### **Summary Table**
| Question ID | Original Question | Best Match Score | Best Match Question | Best Match Answer | Total Matches |
|-------------|-------------------|------------------|---------------------|-------------------|---------------|
| 1 | Does your company... | 0.842 | How does Prysmian... | Prysmian places... | 3 |

### **Detailed Table**
| Question ID | Original Question | Match Rank | Similarity Score | Matched Question | Answer |
|-------------|-------------------|------------|------------------|------------------|--------|
| 1 | Does your company... | 1 | 0.842 | How does Prysmian... | Prysmian places... |
| 1 | Does your company... | 2 | 0.756 | What initiatives... | The company has... |

### **Question-Answer Format**
Interactive expandable sections showing:
- **Question Headers**: Full original questions as section titles
- **Color-Coded Matches**: 🟢 High (≥0.7), 🟡 Medium (≥0.4), 🟠 Low (<0.4) similarity
- **Answer Paragraphs**: Full answers in styled containers
- **Display Modes**: All questions, matches only, or top scoring questions

## ⚙️ **Configuration**

### **Default Settings**
- **Q&A Pairs File**: `pdf-qa-generator/output/qa_pairs.json`
- **Top Matches**: 3 per question
- **Similarity Method**: TF-IDF cosine similarity

### **Customization**
- Adjust top-k matches via sidebar slider
- Change Q&A pairs file path in sidebar
- Results automatically update with new settings

## 🛠️ **Technical Details**

### **Dependencies**
- **Streamlit**: Web interface framework
- **Pandas**: Data manipulation and Excel export
- **Docling**: Document parsing (AI-powered)
- **Custom Modules**: `simple_document_parser.py`, `question_answer_matcher.py`

### **Architecture**
- **Frontend**: Streamlit web interface
- **Backend**: Python processing modules
- **Storage**: Temporary file handling for uploads
- **Export**: In-memory file generation for downloads

## 🚨 **Troubleshooting**

### **Common Issues**

1. **Q&A Pairs File Not Found**
   - Check file path in sidebar
   - Ensure file exists and is readable
   - Verify JSON format

2. **Document Processing Fails**
   - Check file format is supported
   - Ensure file is not corrupted
   - Try smaller file size

3. **No Questions Extracted**
   - Document may not contain recognizable questions
   - Try different document or format
   - Check document text quality

4. **Streamlit Not Found**
   - Run: `pip install streamlit`
   - Or use: `python run_streamlit_app.py` (auto-installs)

### **Performance Tips**
- Use smaller documents for faster processing
- Reduce top-k matches for quicker results
- Close browser tab when done to free resources

## 📈 **Usage Examples**

### **ESG Questionnaires**
- Upload ESG compliance questionnaires
- Match questions to sustainability knowledge base
- Export results for audit purposes

### **Survey Analysis**
- Process survey documents
- Find similar questions in database
- Generate response templates

### **Document Standardization**
- Upload various document formats
- Standardize question formats
- Create unified question databases

## 🔄 **Workflow Integration**

The Streamlit app integrates with existing workflow:

```
Document Upload → Question Extraction → Answer Matching → Results Export
     ↓                    ↓                   ↓              ↓
  Web Interface    simple_document_parser   question_answer   JSON/Excel
                                           _matcher          Download
```

Perfect for:
- **Business Users**: Easy web interface, no coding required
- **Analysts**: Interactive data exploration and export
- **Compliance Teams**: Standardized Q&A processing
- **Researchers**: Bulk document analysis

---

🎉 **Ready to process your documents? Run the app and start uploading!**
