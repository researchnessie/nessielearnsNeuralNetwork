import numpy as np
import onnxruntime as rt
import pandas as pd
#====================================================================================================
subject_net = rt.InferenceSession("model.onnx")

input_name = subject_net.get_inputs()[0].name
output_name = subject_net.get_outputs()[0].name

while True:
    i = str(input("prompt: "))
    input_data = np.array([[i]], dtype=object)
    result = subject_net.run([output_name], {input_name: input_data})
    prediction = result[0][0]
    answer = "this is MMK -> AED question" if prediction == 1 else "this is AED to MMK question"
    print(answer)