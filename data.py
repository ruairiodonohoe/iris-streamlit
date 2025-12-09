import pandas as pd
import streamlit as st

@st.cache_data
def load_iris_data(path="./resources/data/iris.csv"):
    df = pd.read_csv(path)
    
    species_map = df[["target", "species"]].drop_duplicates()
    drop_colnames = species_map.columns
    species_map = dict(zip(df["target"], df["species"]))

    original_feature_colnames = [col for col in df.columns if col not in drop_colnames]
    title_feature_colnames = sorted([col.replace("_", " ").replace(" cm", "").title() for col in original_feature_colnames])

    
    return df, original_feature_colnames, title_feature_colnames, species_map