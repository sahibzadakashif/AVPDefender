import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from Pfeature.pfeature import btc_wp, ddr_wp, rri_wp
import os
import numpy as np
from stmol import showmol
import py3Dmol
import requests
from Bio import SeqIO
from io import StringIO
def main():
    # Set the color scheme
    header_color = '#91C788'
    background_color = '#FFFFFF'
    text_color = '#333333'
    primary_color = '#3BB143'
    footer_color = '#017C8C'
    footer_text_color = '#FFFFFF'
    font = 'Arial, sans serif'

    # Set the page config
    st.set_page_config(
        page_title='AVPDefender',
        layout='wide',
        initial_sidebar_state='expanded',
        page_icon='☣',
    )

    # Set the theme
    st.markdown(f"""
    <style>
        .reportview-container {{
            background-color: {background_color};
            color: {text_color};
            font-family: {font};
        }}
        .sidebar .sidebar-content {{
            background-color: {header_color};
            color: {text_color};
        }}
        .stButton > button {{
            background-color: {primary_color};
            color: {background_color};
            border-radius: 12px;
            font-size: 16px;
            padding: 10px 20px;
        }}
        footer {{
            font-family: {font};
            background-color: {footer_color};
            color: {footer_text_color};
        }}
        .header-title {{
            color: {primary_color};
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
        }}
        .header-subtitle {{
            color: {text_color};
            font-size: 20px;
            text-align: center;
            margin-bottom: 30px;
        }}
    </style>
    """, unsafe_allow_html=True)

    # Add university logos to the page
    left_logo, center, right_logo = st.columns([1, 2, 1])
    left_logo.image("LOGO_u.jpeg", width=270)
    right_logo.image("images.png", width=270)

    # Add header with application title and description
    with center:
        st.markdown("<h1 class='header-title'>AVP Defender – A Prediction Server for AVPs Prediction</h1>", unsafe_allow_html=True)
        st.markdown("""
        <p class='header-subtitle'>
        Welcome to AVP Defender; the revolutionary algorithm for the prediction of Antiviral Peptides with exceptional accuracy of 98%. 
        Utilizing advanced Machine Learning, AVP Defender predicts the probability of toxin-degrading activity of the Antiviral Peptides. 
        Explore the future of antiviral peptide discovery and unlock nature's potential with AVP Defender.
        </p>
        """, unsafe_allow_html=True)
if __name__ == "__main__":
    main()

# Load the trained model
model_file = "rf.pkl"  # Ensure this path is correct
model = joblib.load(model_file)

if 'current_seq_idx' not in st.session_state:
    st.session_state.current_seq_idx = 0

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
        raise ValueError("You have entered an invalid sequence. Please check your input.")
    return True

def update(sequence_list):
    pdb_strings = []
    for sequence in sequence_list:
        # Convert the sequence to uppercase for API compatibility
        uppercase_sequence = sequence.upper()

        if not is_valid_sequence(uppercase_sequence):
            st.error(f"Invalid sequence: {sequence}")
            continue

        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
        }
        response = requests.post('https://api.esmatlas.com/foldSequence/v1/pdb/', headers=headers, data=uppercase_sequence, verify=False)
        if response.status_code == 200:
            pdb_string = response.content.decode('utf-8')
            pdb_strings.append(pdb_string)
        else:
            st.error(f"Error with sequence {sequence}: Status code {response.status_code}")
    return pdb_strings

# 3D Structure Prediction Functions
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

def show_next():
    if 'pdb_strings' in st.session_state:
        st.session_state.current_seq_idx = (st.session_state.current_seq_idx + 1) % len(st.session_state.pdb_strings)
        render_current_structure()


def show_previous():
    if 'pdb_strings' in st.session_state:
        st.session_state.current_seq_idx = (st.session_state.current_seq_idx - 1) % len(st.session_state.pdb_strings)
        render_current_structure()

def render_current_structure():
    if 'pdb_strings' in st.session_state and st.session_state.pdb_strings:
        current_pdb = st.session_state.pdb_strings[st.session_state.current_seq_idx]
        with structure_container:
            # Displaying the index of the current structure
            st.markdown(f"**Displaying Structure {st.session_state.current_seq_idx + 1} of {len(st.session_state.pdb_strings)}**")

            render_mol(current_pdb)

            # Adding a download button for the current structure
            st.download_button(
                label="Download this Structure",
                data=current_pdb,
                file_name=f"structure_{st.session_state.current_seq_idx + 1}.pdb",
                mime='chemical/x-pdb'
            )

# Function to parse FASTA format
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
    btc_df_list = [btc(seq) for seq in sequences if seq]
    ddr_df_list = [ddr(seq) for seq in sequences if seq]
    rri_df_list = [rri(seq) for seq in sequences if seq]
    df_features = pd.concat([pd.concat(btc_df_list, axis=0),
                             pd.concat(ddr_df_list, axis=0),
                             pd.concat(rri_df_list, axis=0)], axis=1)
    feature_cols = ['BTC_T', 'BTC_H', 'BTC_S', 'BTC_D', 'DDR_A', 'DDR_C', 'DDR_D', 'DDR_E', 'DDR_F', 'DDR_G', 'DDR_H', 'DDR_I', 'DDR_K', 'DDR_L', 'DDR_M', 'DDR_N', 'DDR_P', 'DDR_Q', 'DDR_R', 'DDR_S', 'DDR_T', 'DDR_V', 'DDR_W', 'DDR_Y', 'RRI_A', 'RRI_C', 'RRI_D', 'RRI_E', 'RRI_F', 'RRI_G', 'RRI_H', 'RRI_I', 'RRI_K', 'RRI_L', 'RRI_M', 'RRI_N', 'RRI_P', 'RRI_Q', 'RRI_R', 'RRI_S', 'RRI_T', 'RRI_V', 'RRI_W', 'RRI_Y']
    df_features = df_features.reindex(columns=feature_cols, fill_value=0)
    y_pred = model.predict(df_features)
    prediction_probability = model.predict_proba(df_features)[:, 1]
    return y_pred, prediction_probability

# HTML and CSS to color the title
st.markdown(
    """
    <style>
    .title {
        color: #3BB143;  /* Parrot Green color code */
        font-size: 2em;
        font-weight: bold;
    }
    </style>
    <h1 class="title">Enzyme Sequence Submission</h1>
    """,
    unsafe_allow_html=True
)


if 'page' not in st.session_state:
    st.session_state.page = 'input'
if 'submit_count' not in st.session_state:
    st.session_state.submit_count = 0

if st.session_state.page == 'input':
    st.subheader("Please Enter Enzyme Sequences in FASTA Format")
    protein_sequences = st.text_area("Enzyme Sequences (Enter multiple sequences separated by new lines)", height=150)
    fasta_file = st.file_uploader("Or upload FASTA file", type=["fasta", "txt"])

    submit_button = st.button("Submit", key="input_submit")

    if submit_button:
        st.session_state.submit_count += 1

    if fasta_file:
        fasta_content = fasta_file.getvalue().decode("utf-8").splitlines()
        protein_sequences = parse_fasta(fasta_content)
        st.info("File uploaded. Ready to submit.")
    else:
        protein_sequences = protein_sequences.strip().split('\n')

    if submit_button:
        if not protein_sequences:
            st.error("Please enter protein sequences or upload a FASTA file.")
        else:
            st.session_state.protein_sequences = protein_sequences
            predictions, prediction_probability = predict_peptide_structure(protein_sequences)
            st.session_state.prediction = predictions
            st.session_state.prediction_probability = prediction_probability
            st.session_state.page = 'output'

if st.session_state.page == 'output':
    st.subheader("Prediction Results")

    results_df = pd.DataFrame({
        'Index': range(1, len(st.session_state.protein_sequences) + 1),
        'Peptide Sequence': st.session_state.protein_sequences,
        'Predicted Probability': st.session_state.prediction_probability,
        'Class Label': st.session_state.prediction
    })

    st.table(results_df)

    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name='prediction_results.csv',
        mime='text/csv',
    )

    st.button("Back", on_click=lambda: setattr(st.session_state, 'page', 'input'))
    structure_container = st.container()

# HTML and CSS to color the title and header
st.markdown(
    """
    <style>
    .title {
        color: #3BB143;  /* Parrot Green color code */
        font-size: 2em;
        font-weight: bold;
    }
    .header {
        color: #3BB143;  /* Parrot Green color code */
        font-size: 1.5em;
        font-weight: bold;
    }
    </style>
    <h1 class="title">ToxZyme Developers:</h1>
    """,
    unsafe_allow_html=True
)

# Define columns for the profiles
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

with col1:
    # st.image("my-photo.jpg", width=100)
    st.markdown("""
        <div style='line-height: 1.1;'>
            <h3>Dr. Kashif Iqbal Sahibzada</h3>
             Assistant Professor | Department of Health Professional Technologies, Faculty of Allied Health Sciences, The University of Lahore<br>
            Post-Doctoral Fellow | Henan University of Technology,Zhengzhou China<br>
            Email: kashif.iqbal@dhpt.uol.edu.pk | kashif.iqbal@haut.edu.cn
        </div>
    """, unsafe_allow_html=True)

with col2:
    # st.image("colleague-photo.jpg", width=100)
    st.markdown("""
        <div style='line-height: 1.1;'>
            <h3>Dr.Dong Qing Wei</h3>
            Professor<br>
            Shanghai Jiao Tong University, China
            </div>
        """, unsafe_allow_html=True)

with col3:
    # st.image("teacher-photo.jpg", width=100)
    st.markdown("""
        <div style='line-height: 1.1;'>
            <h3 style='color:#006a4e;'>Dr. Munawar Abbas</h3>
                PhD Biological Sciences<br>
                Henan University of Technology, Zhengzhou, China
            </div>
        """, unsafe_allow_html=True)


