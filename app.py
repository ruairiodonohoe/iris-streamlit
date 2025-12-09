import streamlit as st
import pandas as pd
from predict import predict
from data import load_iris_data

df, original_feature_colnames, title_feature_colnames, species_map = load_iris_data()


st.title("Classifying Iris Flowers")
st.markdown("Toy model to play to classify iris flowers into setosa, versicolor, virginica")

st.header("Plant Features")
col1, col2 = st.columns(2)

with col1:
    st.text("Petal Characteristics")
    petal_l = st.slider(title_feature_colnames[0], min_value=1.0, max_value=8.0, step=0.5)
    petal_w = st.slider(title_feature_colnames[1], min_value=2.0, max_value=4.4, step=0.5)
with col2:
    st.text("Sepal Characteristics")
    sepal_l = st.slider(title_feature_colnames[2], min_value=1.0, max_value=7.0, step=0.5)
    sepal_w = st.slider(title_feature_colnames[3], min_value=0.1, max_value=2.5, step=0.5)

if st.button("Predict type of Iris"):
    data = [[petal_l, petal_w, sepal_l, sepal_w]]
    sample_df = pd.DataFrame(data, columns=original_feature_colnames)
    st.dataframe(sample_df)
    result = predict(sample_df)[0]
    result = species_map[result].title()
    st.text(result)