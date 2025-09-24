
#!/bin/bash
cd "$(dirname "$0")"
pip install -r requirements.txt
python run_streamlit_azure.py
