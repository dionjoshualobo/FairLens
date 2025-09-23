import pandas as pd
import requests
import json

# Test data
test_data = {
    'name': ['Alice', 'Bob', 'Carol', 'David'],
    'age': [25, 30, 35, 28],
    'salary': [50000, 60000, 55000, 52000],
    'gender': ['F', 'M', 'F', 'M']
}

df = pd.DataFrame(test_data)
print("✅ Test dataset created")
print(df.head())

# Save to CSV
df.to_csv('test_dataset.csv', index=False)
print("✅ Test dataset saved as CSV")
