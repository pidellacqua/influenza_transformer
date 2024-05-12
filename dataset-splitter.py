import pandas as pd
from sklearn.model_selection import train_test_split

# Leggi il file CSV
file_path = "./data/dfs_merged_upload-original.csv"
data = pd.read_csv(file_path)

# Suddividi i dati in training e test set
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Scrivi i dati di training in un file CSV
train_data.to_csv("./data/train_dataset.csv", index=False)

# Scrivi i dati di test in un file CSV
test_data.to_csv("./data/test_dataset.csv", index=False)

print("Training dataset salvato come 'train_dataset.csv'")
print("Test dataset salvato come 'test_dataset.csv'")
