import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from Pfeature.pfeature import dpc_wp
import os
import numpy as np
from stmol import showmol
import py3Dmol
import requests
from Bio import SeqIO
from io import StringIO

def load_model():
    model_file = "rf.pkl"
    if not os.path.exists(model_file):
        st.error("Model file 'rf.pkl' not found!")
        st.stop()
    return joblib.load(model_file)

def main():
    # Theme and layout settings
    st.set_page_config(page_title='AVPDefender', layout='wide', initial_sidebar_state='expanded', page_icon='☣')

    # Load model
    st.session_state.model = load_model()

    # Set theme CSS
    st.markdown("""
    <style>
        .reportview-container { background-color: #FFFFFF; color: #333333; font-family: Arial, sans serif; }
        .sidebar .sidebar-content { background-color: #91C788; }
        .stButton > button { background-color: #3BB143; color: #FFFFFF; border-radius: 12px; font-size: 16px; padding: 10px 20px; }
        footer { background-color: #017C8C; color: #FFFFFF; font-family: Arial, sans serif; }
        .header-title { color: #3BB143; font-size: 36px; font-weight: bold; text-align: center; margin-top: 20px; }
        .header-subtitle { color: #333333; font-size: 20px; text-align: center; margin-bottom: 30px; }
    </style>
    """, unsafe_allow_html=True)

    # Logos
    left_logo, center, right_logo = st.columns([1, 2, 1])
    left_logo.image("LOGO_u.jpeg", width=270)
    right_logo.image("images.png", width=270)

    # Header
    with center:
        st.markdown("<h1 class='header-title'>AVP Defender – A Prediction Server for AVPs Prediction</h1>", unsafe_allow_html=True)
        st.markdown("""
        <p class='header-subtitle'>
        Welcome to AVP Defender; the revolutionary algorithm for the prediction of Antiviral Peptides with exceptional accuracy of 98%. 
        Utilizing advanced Machine Learning, AVP Defender predicts the probability of toxin-degrading activity of the Antiviral Peptides. 
        Explore the future of antiviral peptide discovery and unlock nature's potential with AVP Defender.
        </p>
        """, unsafe_allow_html=True)

    sequence_submission()
    developer_section()

def dpc(input_seq):
    input_file = 'input_seq.txt'
    output_file = 'output_btc.csv'
    with open(input_file, 'w') as f:
        f.write(">input_sequence\n" + input_seq)
    dpc_wp(input_file, output_file, 1)
    df = pd.read_csv(output_file)
    os.remove(input_file)
    os.remove(output_file)
    return df

def is_valid_sequence(sequence):
    valid_amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    if not sequence or not all(char.upper() in valid_amino_acids for char in sequence):
        raise ValueError("Invalid sequence. Please check your input.")
    return True

def render_mol(pdb):
    if not pdb.strip():
        st.error("Empty PDB data, cannot render.")
        return
    pdbview = py3Dmol.view()
    pdbview.addModel(pdb, 'pdb')
    pdbview.setStyle({'cartoon': {'color': 'spectrum'}})
    pdbview.setBackgroundColor('white')
    pdbview.zoomTo()
    pdbview.zoom(2, 800)
    pdbview.spin(True)
    showmol(pdbview, height=500, width=800)

def parse_fasta(file_content):
    sequences = []
    current_sequence = ""
    for line in file_content:
        if line.startswith('>'):
            if current_sequence:
                sequences.append(current_sequence)
                current_sequence = ""
        else:
            current_sequence += line.strip()
    if current_sequence:
        sequences.append(current_sequence)
    return sequences

def predict_peptide_structure(sequences):
    model = st.session_state.model
    btc_df_list = [dpc(seq) for seq in sequences if seq]
    df_features = pd.concat(btc_df_list, axis=0)
    y_pred = model.predict(df_features)
    prediction_probability = model.predict_proba(df_features)[:, 1]
    return y_pred, prediction_probability

def sequence_submission():
    st.markdown("""
    <style>.title { color: #3BB143; font-size: 2em; font-weight: bold; }</style>
    <h1 class="title">Enzyme Sequence Submission</h1>
    """, unsafe_allow_html=True)

    if 'page' not in st.session_state:
        st.session_state.page = 'input'
    if 'submit_count' not in st.session_state:
        st.session_state.submit_count = 0

    if st.session_state.page == 'input':
        st.subheader("Please Enter Enzyme Sequences in FASTA Format")
        text_input = st.text_area("Enzyme Sequences (one per line)", height=150)
        fasta_file = st.file_uploader("Or upload FASTA file", type=["fasta", "txt"])
        submit_button = st.button("Submit")

        if fasta_file:
            content = fasta_file.getvalue().decode("utf-8").splitlines()
            sequences = parse_fasta(content)
            st.info("File uploaded. Ready to submit.")
        else:
            sequences = text_input.strip().split('\n') if text_input else []

        if submit_button:
            if not sequences:
                st.error("Please enter sequences or upload a file.")
            else:
                st.session_state.protein_sequences = sequences
                predictions, probs = predict_peptide_structure(sequences)
                st.session_state.prediction = predictions
                st.session_state.prediction_probability = probs
                st.session_state.page = 'output'

    if st.session_state.page == 'output':
        st.subheader("Prediction Results")
        df = pd.DataFrame({
            'Index': range(1, len(st.session_state.protein_sequences) + 1),
            'Peptide Sequence': st.session_state.protein_sequences,
            'Predicted Probability': st.session_state.prediction_probability,
            'Class Label': st.session_state.prediction
        })
        st.table(df)
        csv = df.to_csv(index=False)
        st.download_button("Download CSV", data=csv, file_name="prediction_results.csv", mime="text/csv")
        st.button("Back", on_click=lambda: setattr(st.session_state, 'page', 'input'))

def developer_section():
    st.markdown("""
    <style>.title { color: #3BB143; font-size: 2em; font-weight: bold; }</style>
    <h1 class="title">ToxZyme Developers:</h1>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        st.markdown("""
            <h3>Dr. Kashif Iqbal Sahibzada</h3>
            Assistant Professor, University of Lahore<br>
            Postdoc Fellow, Henan University of Technology, China<br>
            Email: kashif.iqbal@dhpt.uol.edu.pk
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <h3>Dr. Dong Qing Wei</h3>
            Professor<br>
            Shanghai Jiao Tong University, China
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <h3 style='color:#006a4e;'>Dr. Munawar Abbas</h3>
            PhD Biological Sciences<br>
            Henan University of Technology, Zhengzhou, China
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
