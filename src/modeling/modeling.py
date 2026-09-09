import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.preprocessing import preprocessing

df = preprocessing.preprocess()

print(df.describe())