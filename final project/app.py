from pathlib import Path
import os
import runpy


APP_DIR = Path(__file__).resolve().parent / "complaint_register" / "complaint_register"


if __name__ == "__main__":
    os.chdir(APP_DIR)
    runpy.run_path(str(APP_DIR / "app.py"), run_name="__main__")
