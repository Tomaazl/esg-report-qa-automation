#!/usr/bin/env python3
"""
Streamlit App Launcher
Simple script to launch the Q&A document processor web app
"""

import subprocess
import sys
import os

def check_streamlit():
    """Check if streamlit is installed"""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def install_streamlit():
    """Install streamlit if not available"""
    print("📦 Installing Streamlit...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
        print("✅ Streamlit installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install Streamlit")
        return False

def main():
    """Launch the Streamlit app"""
    print("🚀 ESG Q&A Document Processor")
    print("=" * 40)
    
    # Check if streamlit is available
    if not check_streamlit():
        print("⚠️  Streamlit not found. Installing...")
        if not install_streamlit():
            print("❌ Cannot proceed without Streamlit")
            return
    
    # Check if the main app file exists
    app_file = "streamlit_qa_app.py"
    if not os.path.exists(app_file):
        print(f"❌ App file not found: {app_file}")
        return
    
    print("🌐 Launching Streamlit app...")
    print("📝 The app will open in your default web browser")
    print("🛑 Press Ctrl+C to stop the app")
    print()
    
    # Launch streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", app_file,
            "--server.headless", "false",
            "--server.runOnSave", "true",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error running app: {str(e)}")

if __name__ == "__main__":
    main()
