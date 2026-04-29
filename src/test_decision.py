import pandas as pd

df = pd.read_csv("data/final_output.csv")

print("\n📊 Action Distribution:\n")
print(df["action"].value_counts())

print("\n📊 Risk Summary:\n")
print(df["risk_score"].describe())