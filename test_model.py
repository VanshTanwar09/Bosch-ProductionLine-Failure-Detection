import numpy as np
from xgboost import XGBClassifier

model = XGBClassifier()
model.load_model('bosch_model.json')

# Normal
test_data = np.zeros(200)
test_data[0] = 100
test_data[1] = 75
test_data[2] = 60

risk = model.predict_proba([test_data])[0,1] * 100
print(f"Normal: {risk:.2f}%")

# Danger
test_data2 = np.zeros(200)
test_data2[0] = 160
test_data2[1] = 120
test_data2[2] = 90

risk2 = model.predict_proba([test_data2])[0,1] * 100
print(f"Danger: {risk2:.2f}%")
