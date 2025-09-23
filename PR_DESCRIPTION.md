# Pull Request: Add Q&A Pair Selection Feature to Streamlit Dashboard

## Description

This PR adds a new feature to the Streamlit dashboard that allows users to select specific question-answer pairs from the knowledge base JSON file. Only the selected Q&A pairs are used for matching questions extracted from uploaded documents.

## Key Features

### 1. New Knowledge Base Tab
- Added a dedicated tab for managing Q&A pairs from the knowledge base
- Users can load any JSON file containing Q&A pairs by specifying the file path
- Visual feedback shows the number of loaded Q&A pairs

### 2. Q&A Pair Selection Interface
- Interactive table displaying all Q&A pairs with checkboxes
- Each row shows:
  - Checkbox for selection
  - Question (truncated for readability)
  - Answer (truncated for readability)
  - Index number

### 3. Selection Controls
- **Select All**: Selects all Q&A pairs
- **Deselect All**: Clears all selections
- **Invert Selection**: Inverts the current selection
- Real-time counter showing selected/total Q&A pairs

### 4. Search and Filter
- Search functionality to filter Q&A pairs by keywords
- Searches both questions and answers
- Maintains selection state while filtering

### 5. Updated Matching Logic
- Modified `match_questions_to_answers` function to accept selected indices
- Only uses selected Q&A pairs for matching
- Shows informational message about the number of selected pairs being used

### 6. Session State Management
- Persistent selection state across tab switches
- Stores:
  - Selected Q&A pair indices
  - Loaded Q&A pairs data
  - Current file path

### 7. Default Behavior
- All Q&A pairs are selected by default when first loaded
- Users must explicitly deselect pairs they don't want to use

## Technical Changes

- Modified `streamlit_qa_app.py`:
  - Added new session state variables for Q&A pair management
  - Created `render_qa_selection_page()` function for the Knowledge Base tab
  - Created `render_qa_matching_page()` function for the Q&A Matching tab
  - Updated `match_questions_to_answers()` to filter Q&A items based on selection
  - Reorganized UI using Streamlit tabs
  - Removed old sidebar configuration in favor of the new tab-based approach

## How to Use

1. **Load Knowledge Base**: 
   - Go to the "Knowledge Base" tab
   - Enter the path to your Q&A pairs JSON file (default: `pdf-qa-generator/output/qa_pairs.json`)
   - Click "Load Q&A Pairs"

2. **Select Q&A Pairs**:
   - Use checkboxes to select/deselect individual pairs
   - Use "Select All", "Deselect All", or "Invert Selection" buttons for bulk operations
   - Use the search box to filter Q&A pairs by keywords

3. **Process Documents**:
   - Go to the "Q&A Matching" tab
   - Upload your document (PDF, Excel, Word, PowerPoint)
   - Click "Process Document"
   - The system will only use the selected Q&A pairs for matching

## Testing

The implementation has been tested with the sample Q&A pairs file at `pdf-qa-generator/output/qa_pairs.json` and works as expected.

## Screenshots

The new interface features:
1. A Knowledge Base tab where users can load and select Q&A pairs
2. A Q&A Matching tab for document processing with selected pairs
3. Clear visual feedback on the number of selected pairs

## Breaking Changes

None. The dashboard maintains backward compatibility and will work with existing Q&A pair JSON files.

## Future Enhancements

Potential improvements for future iterations:
- Export/import selection presets
- Batch operations on filtered results
- Categories or tags for Q&A pairs
- Visualization of selection statistics
- Pagination for large Q&A pair datasets

## Branch Information

- **Feature Branch**: `feature/qa-pair-selection`
- **Base Branch**: `cursor/add-qa-pair-selection-to-dashboard-bc06`

## Pull Request URL

Create the pull request at: https://github.com/Tomaazl/esg-report-qa-automation/pull/new/feature/qa-pair-selection