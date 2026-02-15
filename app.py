"""WWIW entrypoint.

Ce fichier reste volontairement petit.
L'app FastAPI est construite dans wwiw/app_factory.py
"""

from wwiw import create_app

app = create_app()
