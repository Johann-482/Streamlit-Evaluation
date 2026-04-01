import subprocess
import webbrowser
import time
import sys

# Start Streamlit
subprocess.run(
    [sys.executable, "-m", "streamlit", "run", "evaluate.py"],
    check=True
)

# Wait for server to boot
time.sleep(2)

# Open browser
webbrowser.open("http://localhost:8501")