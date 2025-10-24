## Split the dataset without lekage

import pandas as pd
from src.preproecessing import normalize_text,_redact,lemmetization,remove_stopwords,clean_dataframe
from sklearn.model_selection import GroupShuffleSplit

def shuffle_split(df):
  assert {"title","description","category"}.issubset(df.columns)

  # Build a simple leakage-safe group (title+tag);
  if "group" not in df.columns:
      import re
      def _hash_group(s: str) -> str:
          base = re.sub(r'\d+','0', (s or "")).strip().lower()
          return str(abs(hash(base)) % 10**9)
      df["group"] = (df["title"].fillna("") + " " + df.get("tag","").fillna("")).apply(_hash_group)

  gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
  train_val_idx, test_idx = next(gss.split(df, df["category"], df["group"]))
  train_val = df.iloc[train_val_idx].reset_index(drop=True)
  test_df = df.iloc[test_idx].reset_index(drop=True)

  gss2 = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=43) # ~15% of full for val
  train_idx, val_idx = next(gss2.split(train_val, train_val["category"], train_val["group"]))
  train_df = train_val.iloc[train_idx].reset_index(drop=True)
  val_df = train_val.iloc[val_idx].reset_index(drop=True)

  print(len(train_df), len(val_df), len(test_df))
  print(train_df,val_df,test_df)

  return train_df, val_df, test_df