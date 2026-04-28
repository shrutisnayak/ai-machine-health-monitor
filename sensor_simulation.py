# 1. Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import time

# 2. Simulate data
np.random.seed(42)
normal_temp = np.random.normal(loc=70, scale=5, size=100)
anomaly = np.random.normal(loc=95, scale=2, size=10)
temperature = np.concatenate([normal_temp, anomaly])

# 3. Train model ONLY on normal data
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(normal_temp.reshape(-1, 1))

# 4. (Optional) Static visualization
temp_reshaped = temperature.reshape(-1, 1)
predictions = model.predict(temp_reshaped)

plt.plot(temperature, label="Temperature")
anomalies = np.where(predictions == -1)
plt.scatter(anomalies, temperature[anomalies], color='red', label="Anomaly")
plt.title("AI-Based Anomaly Detection")
plt.legend()
plt.show()

# 5. ✅ ADD REAL-TIME SIMULATION HERE
print("\nStarting Real-Time Monitoring...\n")

for i, temp in enumerate(temperature):
    prediction = model.predict([[temp]])

    if prediction[0] == -1:
        print(f"⚠️ Anomaly detected at Time {i} → Temp: {temp:.2f}°C")
    else:
        print(f"✅ Time {i} → Temp: {temp:.2f}°C Normal")

    time.sleep(0.2)