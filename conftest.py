# conftest.py
# Позволяет pytest видеть пакет src при запуске из корня проекта
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))