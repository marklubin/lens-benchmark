"""Single deploy entrypoint for all LENS benchmark Modal services.

Deploy:
    cd infra/modal && modal deploy deploy_all.py
"""
# Import both servers to register all functions on the same app
from llm_server import *  # noqa: F401,F403
from embedding_server import *  # noqa: F401,F403
