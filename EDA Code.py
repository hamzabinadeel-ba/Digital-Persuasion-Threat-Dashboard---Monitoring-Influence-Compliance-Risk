
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer

# EDA 1.1: Descriptive Statistics and Message Length Analysis 
df = pd.read_excel("phishing_cleaned.xlsx")
# 1) Create word-length feature
df["length_words"] = df["message_text"].astype(str).str.split().str.len()

# 2) Message count per label
label_counts = df["message_label"].value_counts().reset_index()
label_counts.columns = ["message_label", "message_count"]

# 3) Descriptive statistics (word counts only)
desc_stats = (
    df.groupby("message_label")[["length_words"]]
      .agg(["mean", "median", "min", "max", "std"])
      .round(2)   #  round all stats here
)

# Flatten multi-index column names
desc_stats.columns = ['_'.join(col).strip() for col in desc_stats.columns.values]
desc_stats = desc_stats.reset_index()

# 4) Merge all into one final rounded table
overview_combined = pd.merge(label_counts, desc_stats, on="message_label", how="left")

# 5) Display table
print("\n=== DATASET OVERVIEW & WORD-LEVEL STATS (ROUNDED) ===")
print(overview_combined)

# 6) Save as Excel file
output_filename = "dataset_overview.xlsx"
overview_combined.to_excel(output_filename, index=False)

print(f"\nTable saved as: {output_filename}")

# EDA 1.2: Urgency Cue Extraction and Frequency Analysis
# 1.2.1: Urgency Cue Extraction
# 1) Load cleaned dataset
df = pd.read_excel("phishing_cleaned.xlsx")

# 2) Define urgency-related terms
urgency_terms = [
    "urgent",
    "immediately",
    "immediate",
    "now",
    "asap",
    "action required",
    "attention",
    "important",
    "respond now",
    "last chance",
    "final notice",
    "warning",
    "verify",
    "verification",
    "alert"
]

# 3) Build regex pattern (case-insensitive)
pattern = r"(" + "|".join([re.escape(term) for term in urgency_terms]) + r")"

# 4) Create features
df["urgency_count"] = df["message_text"].str.lower().str.count(pattern)
df["is_urgent"] = np.where(df["urgency_count"] > 0, 1, 0)

# 5) Aggregate at label level
urgency_summary = (
    df.groupby("message_label")
      .agg(
          total_messages=("message_text", "count"),
          urgent_messages=("is_urgent", "sum"),
      )
      .reset_index()
)

# 6) Convert proportion to percentage
urgency_summary["prop_urgent"] = (
    (urgency_summary["urgent_messages"] / urgency_summary["total_messages"]) * 100
).round(2)   # round to 2 decimal places


# 8) Reorder columns for clarity
urgency_summary = urgency_summary[[
    "message_label",
    "total_messages",
    "urgent_messages",
    "prop_urgent",          # now a percentage
]]

# 9) Display table
print("\n=== URGENCY CUE EXTRACTION BY LABEL (PERCENTAGES) ===")
print(urgency_summary)

# 10) Save as Excel file
output_filename = "urgency_cue_extraction.xlsx"
urgency_summary.to_excel(output_filename, index=False)

print(f"\nTable saved as: {output_filename}")

# 1.2.2: Urgency Cue Frequency Analysis

# 1) Load cleaned dataset
df = pd.read_excel("phishing_cleaned.xlsx")

# 2) Define the same urgency-related terms used earlier
urgency_terms = [
    "urgent",
    "immediately",
    "immediate",
    "now",
    "asap",
    "action required",
    "attention",
    "important",
    "respond now",
    "last chance",
    "final notice",
    "warning",
    "verify",
    "verification",
    "alert"
]

# 3) Create a column for each urgency term: count of that term in each message
for term in urgency_terms:
    col_name = f"count_{term.replace(' ', '_')}"
    df[col_name] = df["message_text"].str.lower().str.count(re.escape(term))

# 4) Aggregate counts per label
#    For each label, sum the occurrences of each urgency term
agg_cols = [f"count_{term.replace(' ', '_')}" for term in urgency_terms]

urgency_freq_label = (
    df.groupby("message_label")[agg_cols]
      .sum()
      .reset_index()
)

# 5) Optional: sort columns for readability (label first, then term columns)
# Already in desired order: message_label + counts

# 6) Display final frequency table
print("\n=== URGENCY CUE FREQUENCY PER LABEL ===")
print(urgency_freq_label)

# 7) Save as Excel file
output_filename = "urgency_cue_frequency_per_label.xlsx"
urgency_freq_label.to_excel(output_filename, index=False)

print(f"\nTable saved as: {output_filename}") 

# EDA 1.3: Keyword and Phrase Frequency Analysis (Unigrams & Bigrams)
# 1.3.1: Unigram Analysis
# 1) Load cleaned dataset
df = pd.read_excel("phishing_cleaned.xlsx")

# 2) Basic text preprocessing function
def preprocess_text(text):
    text = str(text).lower()
    # Remove non-alphanumeric characters (keep spaces)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["message_text"].apply(preprocess_text)

# 3) Set number of top unigrams
TOP_N = 5   # <<< Updated here

# 4) Compute top unigrams per label
all_rows = []
labels = df["message_label"].unique()

for label in labels:
    subset = df[df["message_label"] == label]["clean_text"]

    if subset.empty:
        continue

    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 1))
    X = vectorizer.fit_transform(subset)

    unigram_counts = X.sum(axis=0).A1
    vocab = vectorizer.get_feature_names_out()

    temp_df = pd.DataFrame({
        "term": vocab,
        "count": unigram_counts
    }).sort_values(by="count", ascending=False).head(TOP_N)

    temp_df.insert(0, "message_label", label)

    all_rows.append(temp_df)

# 5) Combine into one final table
keyword_freq_unigrams = pd.concat(all_rows, ignore_index=True)

# 6) Display final table
print("\n=== TOP 5 KEYWORD FREQUENCIES (UNIGRAMS) BY LABEL ===")
print(keyword_freq_unigrams)

# 7) Save as Excel file
output_filename = "keyword_frequency_unigrams.xlsx"
keyword_freq_unigrams.to_excel(output_filename, index=False)

print(f"\nTable saved as: {output_filename}")


#1.3.2: Bigram Analysis
# 1) Load cleaned dataset
df = pd.read_excel("phishing_cleaned.xlsx")

# 2) Basic text preprocessing function (same as unigrams for consistency)
def preprocess_text(text):
    text = str(text).lower()
    # Remove non-alphanumeric characters (keep spaces)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Normalise multiple spaces to a single space
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["message_text"].apply(preprocess_text)

# 3) Set number of top bigrams
TOP_N = 5

# 4) Compute top bigrams per label
all_rows = []
labels = df["message_label"].unique()

for label in labels:
    subset = df[df["message_label"] == label]["clean_text"]

    if subset.empty:
        continue

    # Bigram vectorizer
    vectorizer_bi = CountVectorizer(stop_words="english", ngram_range=(2, 2))
    X_bi = vectorizer_bi.fit_transform(subset)

    bigram_counts = X_bi.sum(axis=0).A1
    vocab_bi = vectorizer_bi.get_feature_names_out()

    temp_df = pd.DataFrame({
        "bigram": vocab_bi,
        "count": bigram_counts
    }).sort_values(by="count", ascending=False).head(TOP_N)

    temp_df.insert(0, "message_label", label)

    all_rows.append(temp_df)

# 5) Combine all label-specific bigram tables into one
bigram_freq = pd.concat(all_rows, ignore_index=True)

# 6) Display final table
print("\n=== TOP 5 BIGRAMS BY LABEL ===")
print(bigram_freq)

# 7) Save as Excel file for Word/dashboard use
output_filename = "bigram_analysis.xlsx"
bigram_freq.to_excel(output_filename, index=False)

print(f"\nTable saved as: {output_filename}")
