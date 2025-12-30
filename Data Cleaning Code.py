import pandas as pd

df = pd.read_excel("phishing_nlp_dataset.xlsx")

df[['message_text', 'message_label']] = df['Corpus'].str.split('\t', n=1, expand=True)

df = df.drop(columns=['Corpus', 'Labels'])

df = df.dropna(subset=['message_text', 'message_label'])

df['message_text'] = df['message_text'].str.strip()
df['message_label'] = df['message_label'].str.strip()

df.to_excel("phishing_cleaned.xlsx", index=False)

print(df.head())
print(df['message_label'].value_counts())
