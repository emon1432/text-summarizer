---
name: manage_server
description: Starts, restarts, or manages the local Flask web server for the text summarization project.
---

# Manage Server Instructions

When the user asks to "start the server", "run the project", or similar short commands:

1. **Verify Environment**: Ensure the Python virtual environment (`venv`) is present. If it does not exist, inform the user or ask for permission to create it and install `requirements.txt`.
2. **Execute Command**: Use the `run_command` tool to execute `python3 app.py` in the `/home/emonideas/Documents/text-summarizer` workspace.
3. **Notify User**: Let the user know the server is running on port 5000 and can be accessed at `http://127.0.0.1:5000`. Do not modify source code unless specifically requested.
