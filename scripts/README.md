# Interactive Brokers Scripts

These are scripts to interact with the IBKR REST API.
Each script should be mostly isolated, with the exception of common utilities.
While the scripts could be written in any programming language, prefer Python (used with `uv`, the python proj manager)
Prefer a modular architecture, keeping files small.
Use simple code, following best practices and design patterns for more complex cases.

# Auth

Inside `/auth`, one can run auth_status.py to get back auth status
