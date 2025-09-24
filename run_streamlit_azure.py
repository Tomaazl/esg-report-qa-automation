#!/usr/bin/env python3
"""
Streamlit App Launcher for Azure Web App
Script to launch the Q&A document processor web app with Azure-specific configuration
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
    """Launch the Streamlit app with Azure Web App configuration"""
    print("🚀 ESG Q&A Document Processor (Azure - VS Code Deploy)")
    print("=" * 50)
    
    # Set working directory to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"📁 Working directory: {os.getcwd()}")
    
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
        print(f"📁 Current directory contents: {os.listdir('.')}")
        return
    
    # Get port from environment (Azure sets this)
    port = os.environ.get('PORT', '8000')
    
    print("🌐 Launching Streamlit app for Azure Web App...")
    print(f"🔧 Configuration: Port {port}, Address 0.0.0.0")
    print("🛑 Press Ctrl+C to stop the app")
    print()
    
    # Launch streamlit with Azure Web App configuration
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", app_file,
            "--server.port", str(port),
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--server.runOnSave", "false",
            "--browser.gatherUsageStats", "false",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error running app: {str(e)}")
        print(f"📁 Current directory: {os.getcwd()}")
        print(f"📁 Directory contents: {os.listdir('.')}")

if __name__ == "__main__":
    main()
