from Color_Console import *
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
#====================================================================================================
print(ctext("1. Loading training data...", text="green"))
data = pd.read_csv("traning_data.csv")

#clean data
cleaned_data = data.dropna(subset=['message', 'answer'])
x = cleaned_data['message'].astype(str)
y = cleaned_data['answer'].astype(int)

print("2. Building the text pipeline (Vectorization + Classifier)...")
model = Pipeline([
    ('vect', TfidfVectorizer(ngram_range=(1, 2))), # Analyzes single words and word pairs
    ('clf', LogisticRegression(max_iter=1000))
])
print("3. Training....")
model.fit(x, y)
time.sleep(1)

print("4. Training Done, exporting your model...!")

initial_type = [('message', StringTensorType([None, 1]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
print("process finished, shutting down the program!")