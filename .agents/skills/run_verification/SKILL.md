---
name: run_verification
description: Runs the automated verification suite (verify_project.py) to check system integrity, dependencies, and model inference.
---

# Run Verification Instructions

When the user asks to "verify the project", "run tests", "check integrity", or similar short commands:

1. **Verify Python Syntax**: (Optional but recommended) Run `python3 -m py_compile app.py summarizer/*.py` to ensure no syntax errors exist.
2. **Execute Script**: Use the `run_command` tool to execute `python3 verify_project.py`.
3. **Execute Full Inference Test**: If the user wants to test actual summarization generation, use `python3 verify_project.py --run-model`.
4. **Report Results**: Provide a concise summary of the test outputs to the user.
