#PhytoChemX (Tool 1 to Tool 20)

#Final Tool 1 to Tool 20:
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import math
import io
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, AllChem, RDKFingerprint
from rdkit.Chem import rdMolDescriptors
from rdkit.ML.Cluster import Butina
from rdkit.Chem import rdFingerprintGenerator
from io import StringIO
from rdkit.Chem import rdFMCS
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              AdaBoostClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Bidirectional, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.optimizers import Adam
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_hist_gradient_boosting
from rdkit.Chem import AllChem as FingerprintMols
from rdkit.Chem import rdMolDescriptors as FingerprintMols
import requests
from PIL import Image
from io import BytesIO
import schedule
import time
import requests
import threading
import plotly.graph_objects as go
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve
import networkx as nx
from scipy.optimize import curve_fit
import rdkit.Chem.QED as QED
from scipy.integrate import odeint
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE
import pubchempy as pcp
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, classification_report, roc_auc_score, precision_recall_curve, confusion_matrix
import matplotlib.ticker as mticker
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import requests
from io import StringIO
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
import joblib
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, Dropout
import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from io import StringIO
import requests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.DataStructs import FingerprintSimilarity
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import scipy.cluster.hierarchy as sch
from rdkit.Chem import rdFMCS
from io import StringIO
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from rdkit.DataStructs import TanimotoSimilarity
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
import numpy as np
from rdkit import Chem, DataStructs
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
import streamlit as st
import requests
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from io import StringIO
from matplotlib_venn import venn2
import plotly.express as px
import numpy as np
from bs4 import BeautifulSoup  # To parse HTML content
import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import shap
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns
import lime
import lime.lime_tabular
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
#Final15 (Tab1 & Tab2)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdChemReactions, Descriptors, Crippen, rdMolDescriptors
import tempfile
import base64
import re
from io import StringIO, BytesIO
from concurrent.futures import ThreadPoolExecutor
from streamlit_option_menu import option_menu
import rdkit.Chem.rdMolDescriptors as rdm
from rdkit.Chem import rdMolDescriptors
import tempfile
import streamlit as st
import pandas as pd
import numpy as np
import requests  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit import DataStructs
from rdkit.Chem import SDWriter
import matplotlib.pyplot as plt
import base64
import threading
import warnings
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN
import base64
from sklearn.neighbors import NearestNeighbors
import io
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
import joblib
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, Dropout
import os
import joblib

#Tool 1 Functions
# Function to ping the app
def ping_app():
    url = "https://chemgenix.streamlit.app/"
    while True:
        try:
            response = requests.get(url)
            print(f"Pinged {url}, status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to ping {url}: {e}")
        time.sleep(600)  # Ping every 10 minutes

# Helper functions for Tool 2
# Function to calculate Ro5 properties
def calculate_ro5_properties(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan], 
                         index=["molecular_weight", "n_hba", "n_hbd", "logp", "ro5_fulfilled"])
    molecular_weight = Descriptors.ExactMolWt(molecule)
    n_hba = Descriptors.NumHAcceptors(molecule)
    n_hbd = Descriptors.NumHDonors(molecule)
    logp = Descriptors.MolLogP(molecule)
    conditions = [molecular_weight <= 500, n_hba <= 10, n_hbd <= 5, logp <= 5]
    ro5_fulfilled = sum(conditions) >= 3
    return pd.Series([molecular_weight, n_hba, n_hbd, logp, ro5_fulfilled],
                     index=["molecular_weight", "n_hba", "n_hbd", "logp", "ro5_fulfilled"])

# Function to calculate mean and std of dataframe
def calculate_mean_std(dataframe):
    stats = dataframe.describe().T
    stats = stats[["mean", "std"]]
    return stats

# Helper function to scale values by thresholds
def _scale_by_thresholds(stats, thresholds, scaled_threshold):
    stats_scaled = stats.apply(lambda x: x / thresholds[x.name] * scaled_threshold, axis=1)
    return stats_scaled

# Helper function to define radial axes angles
def _define_radial_axes_angles(n_axes):
    x_angles = [i / float(n_axes) * 2 * math.pi for i in range(n_axes)]
    x_angles += x_angles[:1]
    return x_angles

# Function to plot radar
def plot_radar(y, thresholds, scaled_threshold, properties_labels, y_max=None, output_path=None):
    n_axes = len(properties_labels)
    x = _define_radial_axes_angles(n_axes)
    
    y = _scale_by_thresholds(y, thresholds, scaled_threshold)
    y_values = y["mean"].tolist()
    y_values += y_values[:1]
    
    if y_max is None:
        y_max = max(y_values) + 1
    
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    
    # Attractive plotting
    ax.fill(x, [scaled_threshold] * (n_axes + 1), "lightcoral", alpha=0.3)
    ax.plot(x, y_values, "mediumblue", lw=3, ls="-")
    ax.plot(x, [y["mean"].tolist()[i] + y["std"].tolist()[i] for i in range(n_axes)] + [y["mean"].tolist()[0] + y["std"].tolist()[0]], "green", lw=2, ls="--")
    ax.plot(x, [y["mean"].tolist()[i] - y["std"].tolist()[i] for i in range(n_axes)] + [y["mean"].tolist()[0] - y["std"].tolist()[0]], "red", lw=2, ls="-.")

    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(180)
    plt.xticks(x[:-1], properties_labels, fontsize=16)
    plt.ylim(0, y_max)
    
    y_max = int(y_max)
    plt.yticks([i for i in range(1, y_max + 1)], [str(i) if i % 1 == 0 else "" for i in range(1, y_max + 1)], fontsize=16)

    labels = ("mean", "mean + std", "mean - std", "rule of five area")
    ax.legend(labels, loc=(1.1, 0.7), labelspacing=0.3, fontsize=16)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight", transparent=True)

    st.pyplot(plt)
    plt.close()

# Tool 3: Helper functions for Tool 3
def calculate_descriptors(molecules):
    descriptors = {
        'MolWt': molecules.apply(lambda mol: Descriptors.MolWt(mol) if mol else None),
        'TPSA': molecules.apply(lambda mol: Descriptors.TPSA(mol) if mol else None),
        'NumRotatableBonds': molecules.apply(lambda mol: Descriptors.NumRotatableBonds(mol) if mol else None),
        'NumHAcceptors': molecules.apply(lambda mol: Descriptors.NumHAcceptors(mol) if mol else None),
        'NumHDonors': molecules.apply(lambda mol: Descriptors.NumHDonors(mol) if mol else None),
    }
    return pd.DataFrame(descriptors)

def process_file(uploaded_file):
    egfr_data = pd.read_csv(uploaded_file)

    if 'unnamed:_0' in egfr_data.columns:
        egfr_data.rename(columns={'unnamed:_0': 'Index'}, inplace=True)
        egfr_data.set_index('Index', inplace=True)

    # Normalize column names to lowercase
    egfr_data.columns = egfr_data.columns.str.strip().str.lower().str.replace(' ', '_')

    # Look for 'smiles' column with any capitalization
    smiles_column = next((col for col in egfr_data.columns if 'smiles' in col), None)
    if smiles_column is None:
        st.error("The uploaded file must contain a 'smiles' column.")
        return pd.DataFrame()

    egfr_data['molecule'] = egfr_data[smiles_column].apply(Chem.MolFromSmiles)

    descriptor_df = calculate_descriptors(egfr_data['molecule'])

    combined_df = pd.concat([egfr_data, descriptor_df], axis=1)

    return combined_df

#Tool 4 Functions
# Function to calculate similarity metrics
def calculate_similarity(query_fp, fp_list, metric="Tanimoto"):
    if metric == "Tanimoto":
        return DataStructs.BulkTanimotoSimilarity(query_fp, fp_list)
    elif metric == "Dice":
        return DataStructs.BulkDiceSimilarity(query_fp, fp_list)
    elif metric == "Cosine":
        return DataStructs.BulkCosineSimilarity(query_fp, fp_list)
    elif metric == "Sokal":
        return DataStructs.BulkSokalSimilarity(query_fp, fp_list)
    elif metric == "Russel":
        return DataStructs.BulkRusselSimilarity(query_fp, fp_list)
    elif metric == "RogotGoldberg":
        return DataStructs.BulkRogotGoldbergSimilarity(query_fp, fp_list)
    elif metric == "Kulczynski":
        return DataStructs.BulkKulczynskiSimilarity(query_fp, fp_list)
    elif metric == "McConnaughey":
        return DataStructs.BulkMcConnaugheySimilarity(query_fp, fp_list)
    elif metric == "Tversky":
        return DataStructs.BulkTverskySimilarity(query_fp, fp_list, 0.5, 0.5)
    elif metric == "BraunBlanquet":
        return DataStructs.BulkBraunBlanquetSimilarity(query_fp, fp_list)
    elif metric == "Morgan":
        return DataStructs.BulkMorganSimilarity(query_fp, fp_list)
    else:
        raise ValueError("Unknown similarity metric.")

# Function for Tool 4: Load data from uploaded file
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, index_col=0)
        return df
    else:
        st.warning("Please upload a CSV file.")
        return pd.DataFrame()

# Function to display plots
def display_plots(fps_tool3, similarity_tool3):
    # Plot similarity heatmap
    if st.session_state.get("show_heatmap", False):
        similarity_matrix = np.zeros((len(fps_tool3), len(fps_tool3)))
        for i in range(len(fps_tool3)):
            for j in range(i, len(fps_tool3)):
                similarity_matrix[i, j] = similarity_matrix[j, i] = DataStructs.FingerprintSimilarity(fps_tool3[i], fps_tool3[j])
        plot_similarity_heatmap4(similarity_matrix)

    # Plot dendrogram
    if st.session_state.get("show_dendrogram", False):
        distance_matrix = tanimoto_distance_matrix(fps_tool3)
        plot_dendrogram4(distance_matrix)

    # Plot PCA
    if st.session_state.get("show_pca", False):
        plot_pca_tsne(fps_tool3, method="pca")

    # Plot t-SNE
    if st.session_state.get("show_tsne", False):
        plot_pca_tsne(fps_tool3, method="tsne")

    # Plot similarity distribution
    if st.session_state.get("show_distribution", False):
        plot_similarity_distribution(similarity_tool3)

    # Plot enrichment
    if st.session_state.get("show_enrichment", False):
        query_similarity = similarity_tool3[0]  # Assuming query similarity is first in the list
        plot_enrichment(similarity_tool3, query_similarity)

# Function to create molecules from SMILES
def create_molecules(df):
    compounds = []
    for _, chembl_id, smiles in df[["chembl_id", "smiles"]].itertuples():
        compounds.append((Chem.MolFromSmiles(smiles), chembl_id))
    return compounds

# Function to generate fingerprints
def generate_fingerprints(compounds):
    rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=5)
    return [rdkit_gen.GetFingerprint(mol) for mol, idx in compounds]

# Function to display cluster information
def display_cluster_info(clusters, compounds):
    for i, cluster in enumerate(clusters, 1):
        st.write(f"Cluster {i}: {len(cluster)} compounds")
        for idx in cluster:
            st.write(f"- {compounds[idx][1]}")

# Function to plot similarity heatmap
def plot_similarity_heatmap4(similarity_matrix, title="Similarity Heatmap"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, cmap='viridis', annot=False, cbar=True)
    plt.title(title)
    plt.xlabel("Molecule Index")
    plt.ylabel("Molecule Index")
    st.pyplot()

# Function to convert fingerprints to a numeric array
def fingerprints_to_array(fingerprints):
    # Determine the size of the fingerprint bit vector
    size = fingerprints[0].GetNumBits()
    # Convert each fingerprint to a binary vector
    array = np.zeros((len(fingerprints), size), dtype=int)
    for i, fp in enumerate(fingerprints):
        bits = list(fp.GetOnBits())
        array[i, bits] = 1
    return array
    
# Function to plot PCA/TSNE
def plot_pca_tsne(fingerprints, method='pca', n_components=2):
    fingerprints_array = fingerprints_to_array(fingerprints)
    
    if method == 'pca':
        reducer = PCA(n_components=n_components)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components)
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")
    
    try:
        reduced_fps = reducer.fit_transform(fingerprints_array)
        plt.figure(figsize=(10, 8))
        plt.scatter(reduced_fps[:, 0], reduced_fps[:, 1], alpha=0.5)
        plt.title(f"{method.upper()} Plot")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        st.pyplot(plt)
    except Exception as e:
        st.error(f"An error occurred while plotting PCA/TSNE: {str(e)}")

# Function to plot similarity distribution
def plot_similarity_distribution(similarity_scores, title="Similarity Distribution"):
    plt.figure(figsize=(10, 8))
    sns.histplot(similarity_scores, kde=True)
    plt.title(title)
    plt.xlabel("Similarity Score")
    plt.ylabel("Frequency")
    st.pyplot()

# Function to calculate enrichment
def plot_enrichment(similarity_scores, query_similarity, title="Enrichment Plot"):
    plt.figure(figsize=(10, 8))
    random_similarities = np.random.choice(similarity_scores, size=len(similarity_scores))
    plt.hist(similarity_scores, bins=30, alpha=0.7, label='Similarities')
    plt.hist(random_similarities, bins=30, alpha=0.5, label='Random Similarities')
    plt.axvline(query_similarity, color='r', linestyle='--', label='Query Similarity')
    plt.title(title)
    plt.xlabel("Similarity Score")
    plt.ylabel("Frequency")
    plt.legend()
    st.pyplot()


# Function to calculate Tanimoto distance matrix
def tanimoto_distance_matrix(fp_list):
    size = len(fp_list)
    dissimilarity_matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(i + 1, size):
            sim = DataStructs.FingerprintSimilarity(fp_list[i], fp_list[j], metric=DataStructs.TanimotoSimilarity)
            dissimilarity_matrix[i, j] = dissimilarity_matrix[j, i] = 1 - sim
    return dissimilarity_matrix

def plot_dendrogram4(distance_matrix, title="Dendrogram"):
    plt.figure(figsize=(10, 8))
    Z = sch.linkage(squareform(distance_matrix), method='ward')  # Use squareform for condensed distance matrix
    sch.dendrogram(Z)
    plt.title(title)
    plt.xlabel("Compound Index")
    plt.ylabel("Distance")
    st.pyplot(plt)

# Function to create fingerprints from SMILES
def create_fingerprints(smiles_list):
    fingerprints = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fingerprints.append(MACCSkeys.GenMACCSKeys(mol))
        else:
            print(f"Invalid SMILES: {smiles}")
    return fingerprints

# Tool 5: Small Molecule Analysis Functions
# Load molecules from file
def load_molecules5(file):
    """Load data from a CSV file and handle potential errors."""
    try:
        df = pd.read_csv(file)
        df.columns = df.columns.str.lower()  # Convert all column names to lowercase for consistency
        if 'smiles' not in df.columns:
            st.error("CSV file must contain a 'smiles' column.")
            return pd.DataFrame()  # Return an empty DataFrame if 'smiles' column is missing
        return df
    except Exception as e:
        st.error(f"An error occurred while loading the file: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error
        
def analyze_molecules(molecules, threshold=0.8):
    if not molecules:
        st.error("No molecules to process.")
        return None, None, None, None, None, None

    mcs1 = rdFMCS.FindMCS(molecules)
    mcs2 = rdFMCS.FindMCS(molecules, threshold=threshold)
    mcs3 = rdFMCS.FindMCS(molecules, threshold=threshold, ringMatchesRingOnly=True)
    
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, 2) for mol in molecules]
    similarity_matrix = np.zeros((len(fingerprints), len(fingerprints)))
    for i, fp1 in enumerate(fingerprints):
        for j, fp2 in enumerate(fingerprints):
            similarity_matrix[i, j] = DataStructs.FingerprintSimilarity(fp1, fp2)
    
    diversity_score = np.mean(similarity_matrix)
    
    substructure_freq = {}
    for mol in molecules:
        for sub in Chem.GetMolFrags(mol, asMols=True):
            smarts = Chem.MolToSmarts(sub)
            substructure_freq[smarts] = substructure_freq.get(smarts, 0) + 1
    
    return mcs1, mcs2, mcs3, similarity_matrix, diversity_score, substructure_freq

def create_molecules5(df):
    """Create RDKit molecules from a DataFrame."""
    compounds = []
    
    # Check if the 'smiles' column exists
    if 'smiles' not in df.columns:
        st.error("CSV file must contain a 'smiles' column.")
        return []

    # Generate dummy IDs if 'chembl_id' is missing
    if 'chembl_id' not in df.columns:
        df['chembl_id'] = ['compound_' + str(i) for i in range(len(df))]  

    for _, row in df.iterrows():
        smiles = row.get('smiles', None)
        chembl_id = row.get('chembl_id', None)
        
        if pd.notnull(smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                compounds.append((mol, chembl_id))
            else:
                st.warning(f"Invalid SMILES string: {smiles}")
        else:
            st.warning("Empty or missing SMILES string found.")

    return compounds
    
# Function to generate fingerprints
def generate_fingerprints5(compounds):
    """Generate fingerprints from RDKit molecules."""
    rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=5)
    fingerprints = []
    for mol, _ in compounds:
        if mol is not None:
            fp = rdkit_gen.GetFingerprint(mol)
            fingerprints.append(fp)
        else:
            st.warning("A molecule could not be converted to a fingerprint.")
    return fingerprints

# Function to convert fingerprints to a numeric array
def fingerprints_to_array5(fingerprints):
    # Determine the size of the fingerprint bit vector
    size = fingerprints[0].GetNumBits()
    # Convert each fingerprint to a binary vector
    array = np.zeros((len(fingerprints), size), dtype=int)
    for i, fp in enumerate(fingerprints):
        bits = list(fp.GetOnBits())
        array[i, bits] = 1
    return array

# Function to calculate Tanimoto distance matrix
def tanimoto_distance_matrix5(fp_list):
    size = len(fp_list)
    dissimilarity_matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(i + 1, size):
            sim = DataStructs.FingerprintSimilarity(fp_list[i], fp_list[j], metric=DataStructs.TanimotoSimilarity)
            dissimilarity_matrix[i, j] = dissimilarity_matrix[j, i] = 1 - sim
    return dissimilarity_matrix
    
# Cluster fingerprints
def cluster_fingerprints5(fingerprints, cutoff):
    dist_matrix = tanimoto_distance_matrix5(fingerprints)
    condensed_dist_matrix = squareform(dist_matrix)
    linked = linkage(condensed_dist_matrix, method='average')
    labels = fcluster(linked, cutoff, criterion='distance')
    clusters = [[] for _ in range(max(labels))]
    for idx, label in enumerate(labels):
        clusters[label - 1].append(idx)
    return clusters

# Visualization functions
def show_cluster_size_dist(clusters):
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.set_xlabel("Cluster index")
    ax.set_ylabel("Number of molecules")
    cluster_sizes = [len(c) for c in clusters]
    ax.bar(range(1, len(clusters) + 1), cluster_sizes, color='darkblue', edgecolor='black', lw=1)
    ax.set_title("Cluster Size Distribution")
    st.pyplot(fig)

def plot_similarity_heatmap5(similarity_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, cmap="YlGnBu", annot=True, fmt=".2f")
    plt.title("Tanimoto Similarity Heatmap")
    plt.xlabel("Molecule Index")
    plt.ylabel("Molecule Index")
    st.pyplot()

def plot_dendrogram5(fingerprints):
    distance_matrix = 1 - np.array([DataStructs.FingerprintSimilarity(fp1, fp2) for fp1 in fingerprints for fp2 in fingerprints])
    distance_matrix = distance_matrix.reshape(len(fingerprints), len(fingerprints))
    linked = linkage(distance_matrix, 'ward')
    plt.figure(figsize=(15, 7))
    dendrogram(linked, orientation='top', distance_sort='descending')
    plt.title("Dendrogram")
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")
    st.pyplot()

def plot_pca5(fingerprints, clusters):
    plt.close('all')
    pca = PCA(n_components=2)
    fingerprints_array = fingerprints_to_array5(fingerprints)
    reduced_fps = pca.fit_transform(fingerprints_array)

    # Create a scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.get_cmap('tab10', len(clusters))  # Use a colormap with enough colors

    for i, cluster in enumerate(clusters):
        cluster_points = reduced_fps[cluster]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i + 1}", color=colors(i))

    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.legend(loc='upper right')

    # Save figure to BytesIO
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Use Streamlit to display the figure
    st.image(buf, use_column_width=True)

# Helper functions for Tool 6
# Tool 6: Small Molecule Analysis Functions
def load_molecules(file):
    if file.name.endswith('.sdf'):
        supplier = Chem.ForwardSDMolSupplier(file)
        mols = [mol for mol in supplier if mol is not None]
    elif file.name.endswith('.csv'):
        df = pd.read_csv(file)

        # Strip whitespace and convert columns to lowercase for comparison
        df.columns = df.columns.str.strip().str.lower()

        # Check for 'smiles' column
        if 'smiles' in df.columns:
            mols = [Chem.MolFromSmiles(smiles) for smiles in df['smiles'] if pd.notna(smiles)]
        else:
            st.error("CSV file must contain a 'smiles' column.")
            return []
    else:
        st.error("Unsupported file format. Please upload a .csv or .sdf file.")
        return []
    
    return mols

# Tool 7 & 8
def smiles_to_fp(smiles, method="maccs", n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((n_bits,))  # Return a zero vector if the molecule is invalid
    if method == "maccs":
        return np.array(MACCSkeys.GenMACCSKeys(mol))
    elif method == "morgan2":
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits))
    elif method == "morgan3":
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=n_bits))
    else:
        return np.array(MACCSkeys.GenMACCSKeys(mol))


def model_training_and_validation(model, data):
    train_x, test_x, train_y, test_y = data
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)
    
    accuracy = accuracy_score(test_y, predictions)
    f1 = f1_score(test_y, predictions)
    precision = precision_score(test_y, predictions)
    recall = recall_score(test_y, predictions)
    fpr, tpr, _ = roc_curve(test_y, model.predict_proba(test_x)[:, 1])
    auc_score = auc(fpr, tpr)
    conf_matrix = confusion_matrix(test_y, predictions)
    
    return accuracy, f1, precision, recall, auc_score, fpr, tpr

def build_rnn_model(input_shape):
    model = Sequential()
    model.add(SimpleRNN(64, input_shape=(input_shape, 1), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_nn_model(input_shape):
    model = Sequential()
    model.add(Dense(64, input_dim=input_shape, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(64, 5, activation='relu', input_shape=(input_shape, 1)))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=(input_shape, 1)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_gru_model(input_shape):
    model = Sequential()
    model.add(GRU(64, input_shape=(input_shape, 1)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_bilstm_model(input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(64, input_shape=(input_shape, 1))))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_transformer_model(input_shape):
    model = Sequential()
    model.add(Dense(64, input_dim=input_shape, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_attention_model(input_shape):
    model = Sequential()
    model.add(Dense(64, input_dim=input_shape, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_residual_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(64, 5, activation='relu', input_shape=(input_shape, 1)))
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def deep_learning_model_training(model, train_x, train_y, test_x, test_y):
    model.fit(train_x, train_y, epochs=10, batch_size=32, verbose=0)
    predictions = (model.predict(test_x) > 0.5).astype(int)
    
    accuracy = accuracy_score(test_y, predictions)
    f1 = f1_score(test_y, predictions)
    precision = precision_score(test_y, predictions)
    recall = recall_score(test_y, predictions)
    fpr, tpr, _ = roc_curve(test_y, model.predict(test_x))
    auc_score = auc(fpr, tpr)
    
    return accuracy, f1, precision, recall, auc_score, fpr, tpr

# Function to plot ROC curve with multiple models
def plot_roc_curve07(results):
    plt.figure(figsize=(10, 6))
    for label, metrics in results.items():
        plt.plot(metrics['fpr'], metrics['tpr'], label=f'{label} (AUC = {metrics["auc_score"]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    st.pyplot(plt)

def plot_roc_curve(fpr, tpr, auc_score, label):
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'{label} (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    st.pyplot(plt)

# Function to convert IC50 to pIC50
def convert_ic50_to_pic50(ic50):
    if ic50 <= 0:
        return None
    return -math.log10(ic50 * 1e-9)

# Function to convert DataFrame to CSV for download
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# Tool 9: Function to plot similarity results
def plot_similarity_results(df, id_column):
    if id_column not in df.columns or "Similarity" not in df.columns:
        st.error(f"Required columns '{id_column}' or 'Similarity' are missing in the DataFrame.")
        return

    try:
        # Convert ID column to numerical indices for plotting
        if df[id_column].dtype == 'object':
            df[id_column] = pd.Categorical(df[id_column]).codes

        # Clean Similarity column
        if df["Similarity"].dtype == 'object':
            df["Similarity"] = df["Similarity"].str.extract('(\d+\.?\d*)', expand=False)
            df["Similarity"] = pd.to_numeric(df["Similarity"], errors='coerce')

        # Drop rows with NaN values in the "Similarity" column
        df = df.dropna(subset=["Similarity"])

        # Set the color palette for Seaborn
        sns.set_palette("bright")

        # Scatter plot of similarity scores
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x=id_column, y="Similarity", ax=ax, 
                        color='teal', edgecolor='black', s=100, alpha=0.75)
        ax.set_title('Similarity Scores Scatter Plot', fontsize=16, color='midnightblue', fontweight='bold')
        ax.set_xlabel(id_column, fontsize=14, color='midnightblue', fontweight='bold')
        ax.set_ylabel('Similarity', fontsize=14, color='midnightblue', fontweight='bold')
        ax.grid(True, linestyle='--', color='gray', alpha=0.6)
        st.pyplot(fig)

        # Histogram of similarity scores
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df["Similarity"], bins=20, kde=True, ax=ax, 
                     color='orange', edgecolor='black', alpha=0.85)
        ax.set_title('Histogram of Similarity Scores', fontsize=16, color='darkorange', fontweight='bold')
        ax.set_xlabel('Similarity', fontsize=14, color='darkorange', fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=14, color='darkorange', fontweight='bold')
        ax.grid(True, linestyle='--', color='gray', alpha=0.6)
        st.pyplot(fig)

        # Box plot of similarity scores
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(y=df["Similarity"], ax=ax, color='darkviolet', 
                    fliersize=8, linewidth=2, saturation=0.75)
        ax.set_title('Box Plot of Similarity Scores', fontsize=16, color='darkviolet', fontweight='bold')
        ax.set_ylabel('Similarity', fontsize=14, color='darkviolet', fontweight='bold')
        ax.grid(True, linestyle='--', color='gray', alpha=0.6)
        st.pyplot(fig)

        # Pair plot of similarity scores vs other features
        if df.shape[1] > 2:
            pairplot_fig = sns.pairplot(df[['Similarity'] + [col for col in df.columns if col != 'Similarity']], 
                                        diag_kind='kde', palette='coolwarm')
            st.pyplot(pairplot_fig)
        else:
            st.write('Not enough features for pair plot')

        # Heatmap of correlation matrix
        if df.shape[1] > 2:
            corr_matrix = df.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='Spectral', ax=ax, 
                        linewidths=0.5, linecolor='black', cbar_kws={'shrink': 0.8})
            ax.set_title('Correlation Heatmap', fontsize=16, color='purple', fontweight='bold')
            st.pyplot(fig)
        else:
            st.write('Not enough features for heatmap')

    except Exception as e:
        pass


# Tab 2 - Protein and Compound Data Extraction
def Protein_data_extractor(query= "EGFR"):   
    cmd="""{
    "size": 20,
    "from": 0,
    "_source": [
        "target_chembl_id",
        "pref_name",
        "target_components",
        "target_type",
        "organism",
        "_metadata.related_compounds.count",
        "_metadata.related_activities.count",
        "tax_id",
        "species_group_flag",
        "cross_references"
    ],
    "query": {
        "bool": {
            "must": [
            {
            "query_string": {
                "analyze_wildcard": true,
                "query": "*"
            }
            }
            ],
            "filter": [
            {
            "query_string": {
                "fields": [
                "organism",
                "organism.alphanumeric_lowercase_keyword",
                "organism.eng_analyzed",
                "organism.std_analyzed",
                "organism.ws_analyzed",
                "pref_name",
                "pref_name.alphanumeric_lowercase_keyword",
                "pref_name.eng_analyzed",
                "pref_name.std_analyzed",
                "pref_name.ws_analyzed",
                "target_chembl_id",
                "target_chembl_id.alphanumeric_lowercase_keyword",
                "target_chembl_id.eng_analyzed",
                "target_chembl_id.std_analyzed",
                "target_chembl_id.ws_analyzed",
                "target_components",
                "target_components.*.alphanumeric_lowercase_keyword",
                "target_components.*.eng_analyzed",
                "target_components.*.std_analyzed",
                "target_components.*.ws_analyzed",
                "target_type",
                "target_type.alphanumeric_lowercase_keyword",
                "target_type.eng_analyzed",
                "target_type.std_analyzed",
                "target_type.ws_analyzed",
                "tax_id",
                "tax_id.alphanumeric_lowercase_keyword",
                "tax_id.eng_analyzed",
                "tax_id.std_analyzed",
                "tax_id.ws_analyzed"
                ],"""
    cmd2="""          }
            }
            ]
        }
    },
    "track_total_hits": true,
    "sort": []
    }"""
    query='"query": "'+query+'"'
    query=cmd+query+cmd2
    api_url = "https://www.ebi.ac.uk/chembl/elk/es/chembl_target/_search"
    headers =  {"Content-Type":"application/json"}
    response = requests.post(api_url, data=query,headers=headers)
    data=response.json()
    target_chembl_id=[]
    organism=[]
    pref_name=[]
    target_type=[]

    for i in range(len(data['hits']['hits'])):
        target_chembl_id.append(data['hits']['hits'][i]['_source']['target_chembl_id'])
        organism.append(data['hits']['hits'][i]['_source']['organism'])
        pref_name.append(data['hits']['hits'][i]['_source']['pref_name'])
        target_type.append(data['hits']['hits'][i]['_source']['target_type'])
    Protein={'Chembl id':target_chembl_id,'Organism':organism,'Pref_name':pref_name,'Target type':target_type}
    df = pd.DataFrame(Protein)
    return(df)


def Compounds_data_extractor(chembl_id = "CHMBL203"):
    api_url = "https://www.ebi.ac.uk/chembl/api/data/activity"
    headers =  {"Content-Type":"application/json"}
    params = {
        'target_chembl_id': chembl_id,
        'assay_type': 'B',
        'type': 'IC50',
        'format': 'json',
        'relation':'=',
        'standard_units':'nM',
        'limit': '1000',
    }
    response = requests.get(api_url,params=params)
    data=response.json()
    bioactivities=data['activities']

    num_page=math.floor(data['page_meta']['total_count']/1000)
    for i in range(num_page):
        response=requests.get("https://www.ebi.ac.uk"+data['page_meta']['next'])
        data=response.json()
        bioactivities.extend(data['activities'])

    molecule_chembl_id=[]
    ic50=[]
    units=[]
    smiles=[]
    
    for activity in bioactivities:
        molecule_chembl_id.append(activity['molecule_chembl_id'])
        ic50.append(activity['standard_value'])
        units.append(activity['standard_units'])
        smiles.append(activity['canonical_smiles'])

    Molecules_dict={'Chembl id':molecule_chembl_id,'IC50':ic50,'unit':units,'Smiles':smiles}
    Molecules = pd.DataFrame(Molecules_dict)
    Molecules.dropna(axis=0, how="any", inplace=True)
    Molecules=Molecules.astype({"IC50": "float64"})
    Molecules.drop_duplicates('Chembl id', keep="first", inplace=True)
    Molecules.reset_index(drop=True, inplace=True)
    Molecules["pIC50"] = Molecules.apply(lambda x: convert_ic50_to_pic50(x.IC50), axis=1)
    Molecules.sort_values(by="pIC50", ascending=False, inplace=True)
    Molecules.reset_index(drop=True, inplace=True)
    return(Molecules)

#Tool 10 Functions
# Function to compute molecular descriptors
def compute_descriptors(smiles_list):
    descriptor_data = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            descriptors = [desc(mol) for name, desc in Descriptors.descList]
            descriptor_data.append(descriptors)
    return pd.DataFrame(descriptor_data, columns=[name for name, desc in Descriptors.descList])

# Function to compute distance-based graph signatures (Morgan fingerprints)
def compute_graph_signatures(smiles_list):
    fingerprints = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fingerprint = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            fingerprints.append(np.array(fingerprint))
    return pd.DataFrame(fingerprints)

# Function to perform pharmacokinetic calculations (noncompartmental analysis)
def calculate_pharmacokinetic_scores(smiles_list):
    pk_scores = []
    
    def exponential_decay(t, a, b):
        return a * np.exp(-b * t)
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Simulated concentration-time data
            time_points = np.array([0, 1, 2, 4, 8, 12, 24])  # in hours
            concentrations = np.random.uniform(0.5, 2.0, len(time_points))  # concentrations

            if len(time_points) != len(concentrations):
                st.error("Time points and concentrations must have the same length")
                return []
            
            # Fit the concentration-time data to an exponential decay function
            try:
                params, _ = curve_fit(exponential_decay, time_points, concentrations)
                auc = np.trapz(concentrations, time_points)  # AUC using NumPy's trapz
                
                dose = 10  # mg, assume a standard dose
                clearance = dose / auc
                volume_of_distribution = dose / params[0]  # Simplified, assuming initial concentration
                
                pk_status = "Pass" if clearance < 1 else "Fail"
                
                pharmacophoric_features = {
                    "Hydrogen Bond Donors": Descriptors.NumHDonors(mol),
                    "Hydrogen Bond Acceptors": Descriptors.NumHAcceptors(mol),
                    "Rotatable Bonds": Descriptors.NumRotatableBonds(mol)
                }
                
                pk_scores.append({
                    "AUC": auc,
                    "Clearance": clearance,
                    "Volume_of_Distribution": volume_of_distribution,
                    "PK_Status": pk_status,
                    **pharmacophoric_features
                })
            except Exception as e:
                st.error(f"Error in PK calculation for {smiles}: {e}")
                pk_scores.append({
                    "AUC": None,
                    "Clearance": None,
                    "Volume_of_Distribution": None,
                    "PK_Status": "Error",
                    "Hydrogen Bond Donors": None,
                    "Hydrogen Bond Acceptors": None,
                    "Rotatable Bonds": None
                })
        else:
            st.error(f"Invalid SMILES string: {smiles}")
            pk_scores.append({
                "AUC": None,
                "Clearance": None,
                "Volume_of_Distribution": None,
                "PK_Status": "Invalid",
                "Hydrogen Bond Donors": None,
                "Hydrogen Bond Acceptors": None,
                "Rotatable Bonds": None
            })
    return pk_scores

# Function for hazard identification and dose-response evaluation
#def hazard_identification(smiles_list):
    #hazard_scores = []
    #for smiles in smiles_list:
        #mol = Chem.MolFromSmiles(smiles)
        #if mol:
            # Compute molecular descriptors associated with toxicity
            #tpsa = Descriptors.TPSA(mol)  # Topological Polar Surface Area
            #logp = Descriptors.MolLogP(mol)  # LogP (octanol-water partition coefficient)
            #qed_score = QED.qed(mol)  # QED score, higher scores indicate drug-likeness
            
            # Simple heuristic rules for hazard identification
            #hazard = "Low Risk"
            #if tpsa > 140 or logp > 5 or qed_score < 0.3:
                #hazard = "High Risk"
            
            #hazard_scores.append({
                #"TPSA": tpsa,
                #"LogP": logp,
                #"QED_Score": qed_score,
                #"Hazard": hazard
            #})
    #return hazard_scores

from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

# Initialize PAINS Structural Alert Filter
pains_catalog = FilterCatalog(FilterCatalogParams.FilterCatalogs.PAINS)

# Function to check for Michael acceptors (Reactive Groups)
def has_michael_acceptor(mol):
    michael_acceptors = [
        "O=C\\C=C", "O=C\\C=N", "O=C\\C#N", "O=C\\C#C", "S=C\\C=C"
    ]
    for pattern in michael_acceptors:
        smarts = Chem.MolFromSmarts(pattern)
        if mol.HasSubstructMatch(smarts):
            return "Yes"
    return "No"

# Hazard Identification Function
def hazard_identification(smiles_list):
    hazard_scores = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Compute Molecular Descriptors
            qed_score = QED.qed(mol)
            mol_weight = Descriptors.MolWt(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            num_aromatic_rings = Descriptors.NumAromaticRings(mol)

            # Check for PAINS Structural Alerts
            pains_alert = "Yes" if pains_catalog.HasMatch(mol) else "No"

            # Detect Reactive Functional Groups (Michael Acceptors)
            michael_acceptor = has_michael_acceptor(mol)

            # Hazard Classification Logic
            hazard = "Low Risk"
            if (qed_score < 0.3 or mol_weight > 500 
                or hbd > 5 or hba > 10 or num_aromatic_rings > 4 
                or pains_alert == "Yes" or michael_acceptor == "Yes"):
                hazard = "High Risk"

            hazard_scores.append({
                "SMILES": smiles,
                "QED_Score": qed_score,
                "MolWt": mol_weight,
                "HBD": hbd,
                "HBA": hba,
                "Aromatic_Rings": num_aromatic_rings,
                "PAINS_Alert": pains_alert,
                "Michael_Acceptor": michael_acceptor,
                "Hazard": hazard
            })

    return hazard_scores

# Function for toxicophore scoring
def calculate_toxicophore_scores(smiles_list):
    toxicophore_scores = []
    toxicophores = [
        '[NX3][C](=[O])[#6]',  # Amide
        '[#6][F]',  # Fluorocarbon
        '[#16]',  # Sulfur-containing
        '[#7]C(=O)',  # Carbamate
        '[#8]C=O'  # Carbonyl
    ]
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            score = 0
            for toxicophore in toxicophores:
                if mol.HasSubstructMatch(Chem.MolFromSmarts(toxicophore)):
                    score += 1
            
            # Interpret toxicophore score
            if score == 0:
                risk_level = "Low Risk"
            elif 1 <= score <= 2:
                risk_level = "Moderate Risk"
            elif 3 <= score <= 4:
                risk_level = "High Risk"
            else:
                risk_level = "Very High Risk"
            
            toxicophore_scores.append({
                "Toxicophore Score": score,
                "Risk Level": risk_level
            })
    
    return toxicophore_scores

# Function to calculate ADMET properties
def calculate_admet_properties(smiles_list):
    admet_properties = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Calculate Lipinski's properties
            lipinski_rules = {
                "Molecular Weight": Descriptors.MolWt(mol),
                "LogP": Descriptors.MolLogP(mol),
                "Hydrogen Bond Donors": Descriptors.NumHDonors(mol),
                "Hydrogen Bond Acceptors": Descriptors.NumHAcceptors(mol),
                "Rotatable Bonds": Descriptors.NumRotatableBonds(mol)
            }
            
            lipinski_status = "Pass"
            if lipinski_rules["Molecular Weight"] > 500 or lipinski_rules["LogP"] > 5 or lipinski_rules["Hydrogen Bond Donors"] > 5 or lipinski_rules["Hydrogen Bond Acceptors"] > 10:
                lipinski_status = "Fail"
            
            # Placeholder calculations for ADMET properties
            absorption = "High" if lipinski_rules["LogP"] < 5 else "Low"
            distribution = "Wide" if lipinski_rules["Molecular Weight"] < 500 else "Limited"
            metabolism = "Fast" if lipinski_rules["Hydrogen Bond Donors"] <= 5 else "Slow"
            excretion = "Efficient" if lipinski_rules["Rotatable Bonds"] <= 7 else "Slow"
            toxicity_status = "Low Risk" if lipinski_status == "Pass" else "High Risk"
            
            admet_properties.append({
                "Molecular Weight": lipinski_rules["Molecular Weight"],
                "LogP": lipinski_rules["LogP"],
                "Hydrogen Bond Donors": lipinski_rules["Hydrogen Bond Donors"],
                "Hydrogen Bond Acceptors": lipinski_rules["Hydrogen Bond Acceptors"],
                "Rotatable Bonds": lipinski_rules["Rotatable Bonds"],
                "Lipinski's Status": lipinski_status,
                "Absorption": absorption,
                "Distribution": distribution,
                "Metabolism": metabolism,
                "Excretion": excretion,
                "Toxicity Status": toxicity_status
            })
    return admet_properties

def calculate_admet_properties2(smiles_list):
    admet_properties = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Lipinski's properties
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            h_donors = Descriptors.NumHDonors(mol)
            h_acceptors = Descriptors.NumHAcceptors(mol)
            rot_bonds = Descriptors.NumRotatableBonds(mol)

            # Placeholder calculations for additional ADMET properties
            hia = "High" if logp < 5 else "Low"  # Human intestinal absorption
            hob = "High" if mw < 500 else "Low"  # Human oral bioavailability
            caco2 = "Permeable" if logp < 3 else "Non-permeable"  # Caco-2 permeability
            bbb = "Penetrant" if logp < 2 else "Non-penetrant"  # Blood-brain barrier penetration
            bpb = "Penetrant" if logp < 3 else "Non-penetrant"  # Blood-placenta barrier penetration

            # Placeholder P-glycoprotein analysis
            p_gp_substrate = "Yes" if mw < 500 and logp < 4 else "No"

            # Plasma protein binding (PPB)
            ppb = "High" if logp > 3 else "Low"

            # Metabolism: CYP450 substrate/inhibitor
            cyp2c9_substrate = "Likely" if mw < 500 and h_acceptors < 10 else "Unlikely"
            cyp3a4_inhibitor = "Yes" if logp > 3 else "No"

            # Placeholder pharmacokinetics transporters
            brcp_inhibitor = "Yes" if logp > 3 else "No"

            # Excretion: Half-life and renal clearance
            t_half = "Short" if rot_bonds < 5 else "Long"
            renal_clearance = "Efficient" if logp < 2 else "Slow"

            # Toxicity properties
            organ_toxicity = "Low" if mw < 500 else "High"
            ames_test = "Non-mutagenic" if h_donors < 5 else "Mutagenic"
            carcinogenicity = "Non-carcinogenic" if logp < 4 else "Carcinogenic"

            # Placeholder eco-toxicity
            phytoplankton_toxicity = "Low" if mw < 300 else "High"

            # Biodegradation
            biodegradation = "Fast" if rot_bonds < 5 else "Slow"

            admet_properties.append({
                "Molecular Weight": mw,
                "LogP": logp,
                "Hydrogen Bond Donors": h_donors,
                "Hydrogen Bond Acceptors": h_acceptors,
                "Rotatable Bonds": rot_bonds,
                "HIA": hia,
                "HOB": hob,
                "Caco-2": caco2,
                "BBB Penetration": bbb,
                "BPB Penetration": bpb,
                "P-gp Substrate": p_gp_substrate,
                "PPB": ppb,
                "CYP2C9 Substrate": cyp2c9_substrate,
                "CYP3A4 Inhibitor": cyp3a4_inhibitor,
                "BRCP Inhibitor": brcp_inhibitor,
                "Half-life": t_half,
                "Renal Clearance": renal_clearance,
                "Organ Toxicity": organ_toxicity,
                "Ames Test": ames_test,
                "Carcinogenicity": carcinogenicity,
                "Phytoplankton Toxicity": phytoplankton_toxicity,
                "Biodegradation": biodegradation
            })
    return admet_properties

# Function to calculate toxicity-related properties
def calculate_toxicity_properties(smiles_list):
    toxicity_properties = []
    for smiles in smiles_list:
        # Example of simplified toxicity predictions
        organ_toxicity = "Low"  # Simplified based on structure (this could be replaced by a model)
        herg_inhibition = "No"  # Simplified rule
        mutagenesis = "No"  # Simplified prediction
        carcinogenesis = "No"  # Simplified prediction
        
        toxicity_properties.append({
            "Organ Toxicity": organ_toxicity,
            "hERG Inhibition": herg_inhibition,
            "Mutagenesis": mutagenesis,
            "Carcinogenesis": carcinogenesis
        })
    return toxicity_properties

# Function to calculate CYP450 and Transporter properties
def calculate_cyp_and_transporters(smiles_list):
    cyp_properties = []
    for smiles in smiles_list:
        # Example of simplified CYP450 interaction predictions
        cyp_interaction = "Substrate"  # Simplified prediction
        brcpi_interaction = "Inhibitor"  # Example transporter interaction
        oapti_interaction = "Non-Inhibitor"  # Example
        
        cyp_properties.append({
            "CYP Interaction": cyp_interaction,
            "BRCPi Interaction": brcpi_interaction,
            "OAPT1b1i Interaction": oapti_interaction
        })
    return cyp_properties

# Function to calculate Absorption & Distribution
def calculate_absorption_distribution(smiles_list):
    absorption_distribution = []
    for smiles in smiles_list:
        # Simplified predictions
        absorption = "High" if "O" in smiles else "Low"
        distribution = "Wide" if "C" in smiles else "Limited"
        
        absorption_distribution.append({
            "Absorption": absorption,
            "Distribution": distribution
        })
    return absorption_distribution

# Load and preprocess the ADR dataset
@st.cache_data
def load_and_preprocess_data():
    data_url = "https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/drug_names_SIDER_with_SMILES_Final_Float_Label.csv"
    df = pd.read_csv(data_url)
    df = df.dropna(subset=['SMILES'])  # Drop rows where SMILES are missing

    # Normalize column names to lowercase
    df.columns = map(str.lower, df.columns)

    # Check for 'label', 'pIC50', or 'logp' in a case-insensitive manner
    label_column = next((col for col in df.columns if col in ['label', 'pic50', 'logp']), None)

    if label_column:
        # Determine the dynamic threshold based on the median of the column
        threshold = df[label_column].median()
        #st.write(f"Using dynamic threshold based on the median value: {threshold:.2f} for column {label_column}")

        # Apply dynamic threshold to create a binary label
        df[label_column] = df[label_column].apply(lambda x: 1 if x >= threshold else 0)

    return df

df = load_and_preprocess_data()

# Compute molecular descriptors
def compute_descriptors(smiles_list):
    descriptors = []
    for smile in smiles_list:
        mol = Chem.MolFromSmiles(smile)
        if mol:
            desc = {
                'MolWt': Descriptors.MolWt(mol),
                'NumHDonors': Descriptors.NumHDonors(mol),
                'NumHAcceptors': Descriptors.NumHAcceptors(mol),
                'LogP': Descriptors.MolLogP(mol)
            }
            descriptors.append(desc)
        else:
            # Handle invalid SMILES case
            st.warning(f"Invalid SMILES string: {smile}")
            descriptors.append({
                'MolWt': None,
                'NumHDonors': None,
                'NumHAcceptors': None,
                'LogP': None
            })
    return pd.DataFrame(descriptors)

def train_gradient_boosting_model(df):
    X = df.drop('label', axis=1)
    y = df['label']
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Ensure that the split is stratified
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    return model, X_test, y_test

def predict_adr(smiles_list, model):
    descriptor_df = compute_descriptors(smiles_list)
    graph_signature_df = compute_graph_signatures(smiles_list)  # Ensure this function is defined
    combined_df = pd.concat([descriptor_df, graph_signature_df], axis=1)
    combined_df = combined_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    if hasattr(model, 'feature_names_in_'):
        expected_columns = model.feature_names_in_
    else:
        st.error("Model does not have attribute 'feature_names_in_'. Please verify the model and feature extraction.")
        return pd.DataFrame()
    combined_df = combined_df.reindex(columns=expected_columns, fill_value=0)
    adr_predictions = model.predict(combined_df)
    return adr_predictions

# Define the score interpretation
def interpret_score(score):
    if score >= 9:
        return "The reaction is considered DOUBTFUL"
    elif 5 <= score <= 8:
        return "The reaction is considered POSSIBLE"
    elif 1 <= score <= 4:
        return "The reaction is considered PROBABLE"
    else:
        return "The reaction is considered DEFINITE"
        
# Set Seaborn theme and context for better aesthetics
sns.set_theme(style="whitegrid")
sns.set_context("talk")


#Tool 13 Functions
# Function to fetch gene interactions from STRING API
def fetch_gene_interactions(gene_list):
    gene_string = "%20".join(gene_list)  # URL encode spaces
    url = f"https://string-db.org/api/json/network?identifier={gene_string}&species=9606"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        data = response.json()
        
        edges = [(entry['preferredName_A'], entry['preferredName_B'], entry['score']) for entry in data]
        return edges
    except requests.RequestException as e:
        print(f"Error fetching gene interactions: {e}")
        return []

# Function to fetch drug-target interactions from ZINC API
def fetch_drug_target_interactions(gene_list):
    query_protein = ','.join(gene_list)
    params = {'gene_name': query_protein, 'count': 'all'}
    url = 'https://zinc15.docking.org/activities.txt:substance.zinc_id+substance.purchasability+gene_name+affinity+structure?'
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        file = StringIO(response.text)
        molecules = pd.read_csv(file, sep="\t", header=None)
        molecules.columns = ["ZINC id", "Purchasability", "Protein id", "pKi", "Smiles"]
        return molecules
    except requests.RequestException as e:
        print(f"Error fetching drug-target interactions: {e}")
        return pd.DataFrame()

# Function to fetch enrichment results using g:Profiler API
def fetch_gprofiler_enrichment_results(gene_list):
    base_url = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"
    params = {
        "organism": "hsapiens",
        "query": gene_list,
        "sources": ["GO:BP", "KEGG", "REAC"]  # Restrict to specific databases like GO, KEGG, and Reactome
    }
    
    try:
        response = requests.post(base_url, json=params)
        response.raise_for_status()  # Check for HTTP errors
        
        # Extract the results
        results = response.json()
        df = pd.DataFrame(results['result'])
        
        if df.empty:
            return None
        
        # Sort by adjusted p-value and return all results
        df_sorted = df.sort_values('p_value')
        return df_sorted[['native', 'name', 'p_value', 'source']]  # Return only relevant columns
    except requests.RequestException as e:
        print(f"Error fetching enrichment results: {e}")
        return None

# Function to plot enrichment results with enhanced visualization
def plot_enrichment_results(enrichment_df):
    # Convert p-values to -log10 scale for better visualization
    enrichment_df['-log10(p-value)'] = -np.log10(enrichment_df['p_value'])
    
    # Use Plotly to create a scatter plot
    fig = px.scatter(
        enrichment_df, 
        x='name', 
        y='-log10(p-value)', 
        color='source', 
        size='-log10(p-value)',
        title='Enrichment Analysis Results',
        labels={'name': 'Pathway', '-log10(p-value)': '-log10(p-value)'},
        color_discrete_sequence=px.colors.qualitative.Pastel  # Choose an attractive color palette
    )
    
    # Customize the layout to make it more suitable for research articles
    fig.update_layout(
        font=dict(size=14),
        title=dict(text='Enrichment Analysis Results', font=dict(size=18)),
        xaxis=dict(title=dict(text='Pathway', font=dict(size=16)), tickangle=45),
        yaxis=dict(title=dict(text='-log10(p-value)', font=dict(size=16))),
        legend=dict(font=dict(size=12), orientation="h", xanchor="center", x=0.5, y=-0.1)
    )

    st.plotly_chart(fig)

# Function to plot animated network
def plot_animated_network(G):
    pos = nx.spring_layout(G)  # Calculate positions
    
    # Create edge and node traces
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=2, color='black'),  # Changed color to black
        hoverinfo='none',
        mode='lines'
    )
    
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers+text',
        textposition='bottom center',
        marker=dict(size=40, color='rgba(255, 102, 102, 0.8)', line=dict(width=2, color='rgb(50, 50, 50)'))  # Increased size and added transparency
    )
    
    # Add edges to trace
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.x += (x0, x1, None)
        edge_trace.y += (y0, y1, None)
    
    # Add nodes to trace
    for node in G.nodes():
        x, y = pos[node]
        node_trace.x += (x,)
        node_trace.y += (y,)
        node_trace.text += (node,)
    
    # Create Plotly figure with enhanced animation
    fig = go.Figure()
    fig.add_trace(edge_trace)
    fig.add_trace(node_trace)
    
    # Create animation frames with more variations
    frames = []
    for i in range(1, 11):  # Increased number of frames for more animation
        frame = go.Frame(
            data=[go.Scatter(x=edge_trace.x, y=edge_trace.y, mode='lines', line=dict(width=2, color=f'rgba(255,0,0,{i*0.1})')),  # Changed color to red
                  go.Scatter(x=node_trace.x, y=node_trace.y, mode='markers+text', marker=dict(size=40, color=f'rgba(255,102,102,{i*0.1})'))],  # Increased size and added transparency
            name=f"Frame {i}"
        )
        frames.append(frame)
    
    fig.frames = frames
    
    fig.update_layout(
        updatemenus=[dict(type='buttons', showactive=False, buttons=[dict(label='Play',
                                                                         method='animate',
                                                                         args=[None, dict(frame=dict(duration=300, redraw=True), fromcurrent=True)])])],
        title="Animated Protein-Protein Interaction Network"
    )
    
    return fig


# Function to plot Venn Diagram
def plot_venn_diagram(targets, drugs):
    # Convert to sets to ensure unique values
    targets_set = set(targets)
    drugs_set = set(drugs)

    # Plot Venn diagram for targets and drugs
    plt.figure(figsize=(4, 3))
    venn = venn2(subsets=(len(targets_set - drugs_set), len(drugs_set - targets_set), len(targets_set & drugs_set)),
                 set_labels=('Targets', 'Drugs'))
    plt.title("Overlap of Targets and Drugs")
    st.pyplot(plt)

#Tool 14 Functions
# Load dataset
@st.cache_data
def load_data(file):
    try:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip().str.lower()
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Generate molecular fingerprints
def generate_fingerprints(compound_df):
    # Print the DataFrame shape and column types for debugging
    #st.write("DataFrame Shape:", compound_df.shape)
    #st.write("DataFrame Columns:", compound_df.columns.tolist())
    #st.write("DataFrame Types:", compound_df.dtypes)

    # Check for empty or NaN values in the 'smiles' column
    if compound_df['smiles'].isnull().any():
        st.warning("There are NaN values in the 'smiles' column.")
    
    # Remove rows with NaN in 'smiles'
    compound_df = compound_df.dropna(subset=['smiles'])
    
    # Initialize lists for storing compounds and fingerprints
    compounds = []
    fingerprints = []

    # Iterate over SMILES strings
    for smiles in compound_df['smiles']:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.warning(f"Invalid SMILES string: {smiles}")
            continue  # Skip invalid SMILES strings
        compounds.append(mol)
        fingerprint = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol)
        fingerprints.append(fingerprint)

    # Check if any fingerprints were generated
    if len(fingerprints) == 0:
        st.error("No valid fingerprints were generated.")
        return [], []

    st.write(f"Generated {len(fingerprints)} fingerprints.")
    return compounds, fingerprints

# Prepare data for the model
def prepare_data_for_model(fingerprints, labels):
    # Ensure fingerprints and labels have the same length
    if len(fingerprints) != len(labels):
        raise ValueError("Mismatch between the number of fingerprints and labels.")
        
    X = np.array([np.array(fp) for fp in fingerprints])
    y = labels
    return X, y

# Build the deep learning model
def build_model(input_shape):
    model = Sequential()
    model.add(Dense(128, input_dim=input_shape, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# Train the deep learning model
def train_model(X_train, X_test, y_train, y_test, compound_df, test_indices):
    model = build_model(X_train.shape[1])
    
    # Train the model and capture the history
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    # Evaluate the model
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    st.write(f"Train MSE: {train_mse}")
    st.write(f"Test MSE: {test_mse}")
    st.write(f"Train R^2: {train_r2}")
    st.write(f"Test R^2: {test_r2}")

    # Plotting functions
    plot_training_loss(history)
    plot_predictions(y_test, y_pred_test)

    # Create DataFrame for predictions
    test_df = compound_df.iloc[test_indices].copy()
    test_df['predicted_pic50'] = y_pred_test.flatten()
    
    # Sort the DataFrame by 'predicted_pic50'
    sorted_test_df = test_df[['smiles', 'predicted_pic50']].sort_values(by='predicted_pic50')
    
    st.write("Predictions (sorted by predicted pIC50):")
    st.write(sorted_test_df)
    
    plot_residuals_histogram(y_test, y_pred_test)
    plot_histogram_of_pIC50(y_test, y_pred_test)
    plot_parity_plot(y_test, y_pred_test)
    plot_correlation_matrix(X_train, y_train)

    # Pass the model_type explicitly
    plot_feature_importance(model, X_train, y_train, model_type="Neural Network")

    plot_pIC50_distribution(compound_df)
    plot_residuals(y_test, y_pred_test)

    return model

# Plotting functions
def plot_predictions(y_test, y_pred_test):
    df = pd.DataFrame({
        'Actual pIC50': y_test.flatten(),
        'Predicted pIC50': y_pred_test.flatten()
    })
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='Actual pIC50', y='Predicted pIC50', alpha=0.7, ax=ax, color='black', label='Predicted')
    sns.scatterplot(data=df, x='Actual pIC50', y=df['Actual pIC50'], alpha=0.7, ax=ax, color='red', label='Actual')
    ax.set_xlabel("Actual pIC50")
    ax.set_ylabel("Predicted pIC50")
    ax.set_title("Actual vs Predicted pIC50")
    ax.legend()
    st.pyplot(fig)

def plot_residuals(y_test, y_pred_test):
    residuals = y_test - y_pred_test
    fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the figure size as needed
    sns.histplot(residuals, bins=20, kde=True, ax=ax)
    ax.set_xlabel("Residuals", fontsize=12)  # Set font size for x-axis label
    ax.set_title("Distribution of Residuals", fontsize=14)  # Set font size for title
    ax.tick_params(axis='both', which='major', labelsize=10)  # Set font size for tick labels
    st.pyplot(fig)

def plot_residuals_histogram(y_test, y_pred_test):
    residuals = y_test - y_pred_test
    residuals_flat = residuals.flatten()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(residuals_flat, bins=30, kde=True, ax=ax, color='teal')
    ax.set_xlabel("Residuals", fontsize=14)
    ax.set_title("Histogram of Residuals", fontsize=16)
    ax.grid(True)
    st.pyplot(fig)

def plot_histogram_of_pIC50(y_test, y_pred_test):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(y_test, bins=20, color='darkgray', alpha=0.6, label='Actual', ax=ax)
    sns.histplot(y_pred_test, bins=20, color='coral', alpha=0.6, label='Predicted', ax=ax)
    ax.set_xlabel("pIC50", fontsize=14)
    ax.set_title("Histogram of Actual vs Predicted pIC50", fontsize=16)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def plot_parity_plot(y_test, y_pred_test):
    y_test = np.ravel(y_test)
    y_pred_test = np.ravel(y_pred_test)
    df = pd.DataFrame({
        'Actual pIC50': y_test,
        'Predicted pIC50': y_pred_test
    })
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x='Actual pIC50', y='Predicted pIC50', alpha=0.7, ax=ax, color='royalblue')
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2, label='Perfect Prediction')
    ax.set_xlabel("Actual pIC50", fontsize=14)
    ax.set_ylabel("Predicted pIC50", fontsize=14)
    ax.set_title("Parity Plot", fontsize=16)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def plot_correlation_matrix(X, y):
    corr = np.corrcoef(X, rowvar=False)
    fig, ax = plt.subplots()
    sns.heatmap(corr, ax=ax, cmap="YlGnBu", square=True)
    ax.set_title("Correlation Matrix of Fingerprints")
    st.pyplot(fig)

def plot_training_loss(history):
    fig, ax = plt.subplots()
    if 'loss' in history.history:
        ax.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        ax.plot(history.history['val_loss'], label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    st.pyplot(fig)

def plot_feature_importance(model, X_train, y_train, model_type):
    # Handle feature importance for different model types
    if model_type in ["XGBoost", "Random Forest"]:
        # For tree-based models, you can use their feature importance attribute
        importances = model.feature_importances_
        feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]

        df_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=df_importance, ax=ax, palette='viridis')
        ax.set_title('Feature Importance', fontsize=16)
        st.pyplot(fig)

    elif model_type == "Neural Network":
        # For neural networks, use LIME to explain feature importance
        explainer = lime.lime_tabular.LimeTabularExplainer(X_train, mode='regression')
        explanation = explainer.explain_instance(X_train[0], model.predict)

        df_importance = pd.DataFrame({
            'Feature': [x[0] for x in explanation.as_list()],
            'Importance': [x[1] for x in explanation.as_list()]
        }).sort_values(by='Importance', ascending=False)

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=df_importance, ax=ax, palette='viridis')
        ax.set_title('Feature Importance (LIME)', fontsize=16)
        st.pyplot(fig)

def plot_pIC50_distribution(compound_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(compound_df['pic50'], bins=20, kde=True, ax=ax, color='indigo')
    ax.set_xlabel("pIC50", fontsize=14)
    ax.set_title("pIC50 Distribution", fontsize=16)
    ax.grid(True)
    st.pyplot(fig)

# --- Helper Functions (Tool 15)---

# First script helper functions
# Function to apply chemical transformations
def apply_transformations(smiles_list, reactions_list, max_cycles=10):
    results = []
    for smile in smiles_list:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            results.append({"SMILES": smile, "Result": "Invalid SMILES"})
            continue

        transformed = False
        for rxn_smarts in reactions_list:
            try:
                reaction = rdChemReactions.ReactionFromSmarts(rxn_smarts)
                if reaction is None:
                    results.append({"SMILES": smile, "Result": "Invalid Reaction SMARTS"})
                    continue

                # Run the reaction
                products = reaction.RunReactants([mol])

                if not products:
                    continue  # Try the next reaction

                # Convert products to SMILES
                product_smiles = [Chem.MolToSmiles(prod) for prod in products[0]]
                results.append({"SMILES": smile, "Result": ", ".join(product_smiles)})
                transformed = True
                break  # Exit once the first successful transformation is found
            except Exception as e:
                results.append({"SMILES": smile, "Result": f"Error: {e}"})

        if not transformed:
            results.append({"SMILES": smile, "Result": "No products"})

    return results

# Function to convert SMILES to SDF
def smiles_to_sdf(smiles_list, names):
    sdf_buffer = StringIO()
    writer = Chem.SDWriter(sdf_buffer)

    for smiles, name in zip(smiles_list, names):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mol.SetProp("_Name", name)
            writer.write(mol)

    writer.close()
    sdf_data = sdf_buffer.getvalue().encode('utf-8')  # Encode to bytes
    return sdf_data

# Function to calculate molecular properties
def analyze_molecular_properties(smiles_list):
    analysis_results = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        h_donors = Descriptors.NumHDonors(mol)
        h_acceptors = Descriptors.NumHAcceptors(mol)
        analysis_results.append({
            'SMILES': smiles, 
            'MW': mw, 
            'LogP': logp, 
            'TPSA': tpsa, 
            'Rotatable Bonds': rot_bonds, 
            'H-Bond Donors': h_donors, 
            'H-Bond Acceptors': h_acceptors
        })
    return pd.DataFrame(analysis_results)

# Second script helper functions
# Helper function to validate and sanitize SMILES strings
def validate_smiles(smiles):
    """Validate the SMILES string to ensure it is correct and safe."""
    if not smiles:
        return False
    # Check if the SMILES string contains valid characters only
    pattern = re.compile(r'[^CcNnOoPpSsFfClBrIiHh0-9@+\-=#()%[\]]')
    if pattern.search(smiles):
        return False
    return True

# Generate list of available chemical transformations for users to select
def generate_transformation_options():
    """Provide a list of available chemical transformations for users to select."""
    return [
        'Sulfonamidation', 'Alkylation', 'Amination', 'Aminolysis', 'Acylation', 
        'Carbamoylation', 'Oxidation', 'Reduction', 'Aryl Coupling', 'Esterification',
        'Etherification', 'Nitration', 'Halogenation', 'Cyclization', 'Dehydration',
        'Deprotection', 'Methylation', 'Phosphorylation', 'Hydrogenation', 'Hydrolysis',
        'Transesterification', 'Condensation', 'Hydroamination', 'Hydroformylation',
        'Ring-Closing Metathesis (RCM)', 'Grignard Reaction', 'Friedel-Crafts Alkylation/Acyloylation',
        'Michael Addition', 'Diels-Alder Reaction', 'Cyclopropanation', 'Wittig Reaction',
        'Claisen Condensation', 'Buchwald-Hartwig Coupling', 'Sonogashira Coupling',
        'Suzuki-Miyaura Coupling', 'Stille Coupling', 'Negishi Coupling', 'Silylation',
        'Tosylation', 'Mesylation', 'Fischer Esterification', 'Vilsmeier-Haack Reaction',
        'Pfitzner-Moffatt Oxidation', 'Mitsunobu Reaction', 'Borylation'
    ]

# Perform the selected chemical transformation
def perform_transformation(smiles, reaction_type):
    """Perform the selected chemical transformation."""
    if not validate_smiles(smiles):
        raise ValueError("Invalid SMILES string provided.")

    # Simulate transformations using simple string manipulation
    transformations = {
        'Sulfonamidation': smiles.replace('c', 'cS(=O)(=O)N'),
        'Alkylation': smiles + 'C',
        'Amination': smiles + 'N',
        'Aminolysis': smiles.replace('C(=O)O', 'C(=O)N'),
        'Acylation': smiles + 'C(=O)R',
        'Carbamoylation': smiles + 'C(=O)N',
        'Oxidation': smiles + 'O',
        'Reduction': smiles.replace('=O', ''),
        'Aryl Coupling': smiles.replace('c1', 'c1-c2'),
        'Esterification': smiles.replace('OH', 'OC=O'),
        'Etherification': smiles.replace('OH', 'OR'),
        'Nitration': smiles.replace('c', 'c[N+](=O)[O-]'),
        'Halogenation': smiles.replace('c', 'cCl'),
        'Cyclization': smiles + 'C1CC1',
        'Dehydration': smiles.replace('O', ''),
        'Deprotection': smiles.replace('OC', 'O'),
        'Methylation': smiles + 'C',
        'Phosphorylation': smiles + 'P(=O)(O)O',
        'Hydrogenation': smiles.replace('C=C', 'C-C'),
        'Hydrolysis': smiles.replace('C(=O)O', 'C(=O)OH'),
        'Transesterification': smiles.replace('C(=O)O', 'C(=O)OR'),
        'Condensation': smiles.replace('O', 'OC=O'),
        'Hydroamination': smiles + 'NH2',
        'Hydroformylation': smiles + 'CH2CHO',
        'Ring-Closing Metathesis (RCM)': smiles + 'C1CC1',
        'Grignard Reaction': smiles + 'MgBr',
        'Friedel-Crafts Alkylation/Acyloylation': smiles + 'C(=O)R',
        'Michael Addition': smiles + 'C(=O)N',
        'Diels-Alder Reaction': smiles + 'C1CC2C=CC2C1',
        'Cyclopropanation': smiles + 'C1CC1',
        'Wittig Reaction': smiles.replace('C=O', 'C=PPh3'),
        'Claisen Condensation': smiles + 'C(=O)O',
        'Buchwald-Hartwig Coupling': smiles + 'C1=CC=C1',
        'Sonogashira Coupling': smiles + 'C#C',
        'Suzuki-Miyaura Coupling': smiles + 'C1=CC2C=CC2C1',
        'Stille Coupling': smiles + 'C1=CC=C1',
        'Negishi Coupling': smiles + 'C1=CC=C1',
        'Silylation': smiles + 'SiMe3',
        'Tosylation': smiles + 'C6H5SO3',
        'Mesylation': smiles + 'CH3SO3',
        'Fischer Esterification': smiles + 'C(=O)OCH3',
        'Vilsmeier-Haack Reaction': smiles + 'C=O',
        'Pfitzner-Moffatt Oxidation': smiles + 'C(=O)O',
        'Mitsunobu Reaction': smiles + 'C(O)N',
        'Borylation': smiles + 'B'
    }
    
    return transformations.get(reaction_type, smiles)

# Compute molecular descriptors using RDKit
def compute_molecular_descriptors(smiles):
    """Compute a comprehensive set of molecular descriptors using RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    
    descriptors = {
        'Molecular Weight': Descriptors.MolWt(mol),
        'LogP': Crippen.MolLogP(mol),
        'Num H Donors': Descriptors.NumHDonors(mol),
        'Num H Acceptors': Descriptors.NumHAcceptors(mol),
        'TPSA': Descriptors.TPSA(mol),
        'Fraction Csp3': Descriptors.FractionCSP3(mol),
        'Num Rotatable Bonds': Descriptors.NumRotatableBonds(mol),
        'Heavy Atom Count': Descriptors.HeavyAtomCount(mol),
        'Exact Mass': Descriptors.ExactMolWt(mol),
        'Ring Count': Descriptors.RingCount(mol),
        'Num Aromatic Rings': Descriptors.NumAromaticRings(mol),
        'Num Valence Electrons': Descriptors.NumValenceElectrons(mol),
        'Num Heteroatoms': Descriptors.NumHeteroatoms(mol),
    }
    
    return descriptors

# Lipinski's rule of five
def lipinski_rule(descriptors):
    """Evaluate Lipinski's rule of five."""
    mw = descriptors.get('Molecular Weight', None)
    logp = descriptors.get('LogP', None)
    h_donors = descriptors.get('H-Bond Donors', None)
    h_acceptors = descriptors.get('H-Bond Acceptors', None)
    
    # Check if any required descriptor is missing
    if mw is None or logp is None or h_donors is None or h_acceptors is None:
        return "Incomplete descriptor data"

    rule = (mw <= 500 and
            logp <= 5 and
            h_donors <= 5 and
            h_acceptors <= 10)
    return rule

def compute_sa_score(mol):
    if mol is None:
        return "Invalid molecule"
    
    try:
        # Check if the function exists
        if hasattr(rdMolDescriptors, 'CalcSyntheticAccessibilityScore'):
            sa_score = rdMolDescriptors.CalcSyntheticAccessibilityScore(mol)
            if sa_score is None:
                return "SA Score calculation returned None"
            return sa_score
        else:
            return "SA Score function is not available in this RDKit version"
    except Exception as e:
        return f"Error calculating SA Score: {e}"


# Download link for CSV
def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="analysis_results.csv">Download CSV File</a>'
    return href

#Functions for Tool 16

# Load dataset with caching
@st.cache_data
def load_data16():
    data_path = "https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/PROTACGx/protac_activity_db.csv"
    df = pd.read_csv(data_path)
    return df

def mol_to_fp16(mol):
    mol = Chem.MolFromSmiles(mol)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    return list(fp)

# Predict PROTAC activity for a given SMILES string
def predict_protac_activity(models, scaler, smiles):
    mol_fp = mol_to_fp16(smiles)
    mol_fp_scaled = scaler.transform([mol_fp])
    
    predictions = {}
    for name, model in models.items():
        if name in ["LSTM", "SimpleRNN"]:
            mol_fp_scaled_reshaped = mol_fp_scaled.reshape(mol_fp_scaled.shape[0], 1, mol_fp_scaled.shape[1])
            y_prob = model.predict(mol_fp_scaled_reshaped)[0]
        else:
            y_prob = model.predict(mol_fp_scaled)[0]
        
        predictions[name] = y_prob[0]
    
    return predictions

# Extract molecular properties from a SMILES string
def extract_molecular_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    properties = {}
    properties['Molecular Weight'] = Descriptors.MolWt(mol)
    properties['Number of H-Bond Donors'] = Descriptors.NumHDonors(mol)
    properties['Number of H-Bond Acceptors'] = Descriptors.NumHAcceptors(mol)
    return properties

# Describe PROTAC activity based on probability
def describe_activity(probability):
    return "Active" if probability >= 0.5 else "Inactive"

# Plot bar chart for predicted activity probability
def plot_activity_probability(prob):
    fig, ax = plt.subplots()
    ax.bar(['Inactive', 'Active'], [1 - prob, prob], color=['blue', 'green'])
    ax.set_xlabel('Activity')
    ax.set_ylabel('Probability')
    ax.set_title('Predicted Activity Probability')
    return fig

# Function to compare accuracy with other tools
def compare_model_performance(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        if name in ["LSTM", "SimpleRNN"]:
            # Reshape for LSTM and SimpleRNN input
            X_test_reshaped = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
            accuracy = model.evaluate(X_test_reshaped, y_test, verbose=0)[1]
        else:
            accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
        
        results[name] = accuracy
    
    # Comparison with other public tools
    public_tool_accuracy = {
        "PROflow": 0.78,
        "PROTAC_splitter": 0.80,
        "ROTAC-Degradation-Predictor": 0.72,
        "Deep-PROTAC": 0.61
    }
    
    results.update(public_tool_accuracy)
    return results


# Function to download file as CSV
def get_binary_file_downloader_html(bin_file, file_name):
    bin_data = bin_file.to_csv(index=False).encode()
    bin_str = base64.b64encode(bin_data).decode()
    href = f'<a href="data:file/csv;base64,{bin_str}" download="{file_name}">Download {file_name}</a>'
    return href

from sklearn.metrics import roc_auc_score

def plot_roc_auc_curve(models, X_test, y_test):
    plt.figure(figsize=(10, 6))
    for name, model in models.items():
        if name in ["LSTM", "SimpleRNN"]:
            X_test_reshaped = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
            y_pred = model.predict(X_test_reshaped)
        else:
            y_pred = model.predict(X_test)
        
        roc_auc = roc_auc_score(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid()
    return plt

# Function to generate SDF file from SMILES
from io import StringIO  # Add this import at the top

# Function to convert SMILES to SDF
def smiles_to_sdf(smiles_list, names):
    sdf_buffer = StringIO()
    writer = Chem.SDWriter(sdf_buffer)

    for smiles, name in zip(smiles_list, names):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mol.SetProp("_Name", name)
            writer.write(mol)

    writer.close()
    sdf_data = sdf_buffer.getvalue().encode('utf-8')  # Encode to bytes
    return sdf_data
    
def generate_sdf(smiles_list, filename='generated_protac.sdf'):
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    with Chem.SDWriter(filename) as writer:
        for mol in mols:
            writer.write(mol)

# Function to generate SDF file in-memory
def generate_sdf_in_memory(smiles_list):
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    sdf_buffer = io.BytesIO()  # Use an in-memory buffer
    with Chem.SDWriter(sdf_buffer) as writer:
        for mol in mols:
            if mol is not None:  # Ensure valid molecule objects
                writer.write(mol)
    sdf_buffer.seek(0)  # Reset the buffer position to the beginning
    return sdf_buffer.getvalue()  # Get the binary content of the SDF file
    
# Function to create a download link for the SDF file
def download_sdf(filename):
    with open(filename, "rb") as f:
        sdf_data = f.read()
    b64_sdf = base64.b64encode(sdf_data).decode()
    href = f'<a href="data:file/sdf;base64,{b64_sdf}" download="{filename}">Download {filename}</a>'
    return href

from rdkit import Chem
from rdkit.Chem import Descriptors

def calculate_toxicity_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"SMILES": smiles, "TSP": None, "Other Toxicity Metric": None}
    
    # Example metrics; you can add more as needed
    logp = Descriptors.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    num_aromatic_rings = Descriptors.NumAromaticRings(mol)
    
    # Placeholder for TSP; this should be replaced with an actual calculation or model
    tsp = logp  # Example: using logP as a placeholder for TSP

    return {"SMILES": smiles, "TSP": tsp, "LogP": logp, "Molecular Weight": mw, "Aromatic Rings": num_aromatic_rings}

# Functions for Tool 17

# Load dataset
@st.cache_data
def load_data17():
    url = "https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/NanoMed/Complete_Dataset.xlsx"
    df = pd.read_excel(url)
    return df

# Preprocess dataset
def preprocess_data17(df):
    df['Pgp'] = df['Pgp'].apply(lambda x: 1 if x == 'yes' else 0)
    features = df[['Weight', 'logP', 'Solubility', 'DrugCarrierRatio', 'Size', 'Pgp']]
    df['pKa_Label'] = np.where(df['pKa'] > 4.5, 1, 0)
    target = df['pKa_Label']
    return features, target

# Machine Learning Models
def train_model17(features, target, model_type="XGBoost"):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    if model_type == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    elif model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "SVM":
        model = SVC(probability=True, random_state=42)
    elif model_type == "MLP":
        model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
        
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    return model, accuracy, X_test, y_test, predictions

# Plot Confusion Matrix
def plot_confusion_matrix(y_test, predictions):
    cm = confusion_matrix(y_test, predictions)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax)
    ax.set_title("Confusion Matrix", fontsize=14)
    ax.set_xlabel("Predicted Labels", fontsize=12)
    ax.set_ylabel("True Labels", fontsize=12)
    st.pyplot(fig)

# Feature Importance Plot (for XGBoost and Random Forest)
def plot_feature_importance17(model, features, model_type):
    if model_type in ["XGBoost", "Random Forest"]:
        importance = model.feature_importances_
        fig, ax = plt.subplots()
        sns.barplot(x=importance, y=features.columns, palette="Blues_d", ax=ax)
        ax.set_title('Feature Importance', fontsize=14)
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_ylabel('Features', fontsize=12)
        st.pyplot(fig)

# Plot ROC Curve
def plot_roc_curve17(model, X_test, y_test):
    y_scores = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', label=f'ROC Curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='red', linestyle='--')
    ax.set_title('ROC Curve', fontsize=14)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.legend(loc='lower right')
    st.pyplot(fig)

# Nanoparticle analysis
def nanoparticle_analysis(data):
    st.write("### Nanoparticle-Specific Analysis")
    
    # Analysis based on different nanoparticles
    nanoparticle_types = ['GOLD', 'SPION', 'AU', 'Carbon', 'Dendrimers', 
                          'Metallic nanoparticles', 'Magnetic nanoparticles',
                          'Polymeric nanoparticles', 'Biological nanomaterials', 
                          'Silica nanoparticles']
    
    # Filter data for nanoparticles analysis
    selected_np = st.selectbox("Select Nanoparticle Type", nanoparticle_types)
    
    # Example analysis: size distribution and zeta potential
    if selected_np in data['CarrierNature'].unique():
        filtered_data = data[data['CarrierNature'] == selected_np]
        
        st.write(f"### Size Distribution for {selected_np}")
        fig = px.histogram(filtered_data, x='Size', nbins=20, title=f"Size Distribution of {selected_np} Nanoparticles",
                           labels={'Size': 'Size (nm)'}, template="plotly_dark")
        st.plotly_chart(fig)
        
        st.write(f"### Zeta Potential for {selected_np}")
        fig = px.violin(filtered_data, x="CarrierNature", y="Zeta", title=f"Zeta Potential of {selected_np}",
                        labels={'Zeta': 'Zeta Potential (mV)'}, template="plotly_dark")
        st.plotly_chart(fig)

# Additional Pharmacokinetics Analysis
def pharmacokinetics_analysis(data):
    st.write("### Pharmacokinetics Analysis")
    
    # Drug release vs time
    #st.write("Drug Release vs Time")
    #fig, ax = plt.subplots()
    #sns.lineplot(x="T max of drug release (hour)", y="Maximium release (%)", data=data, color='blue', ax=ax)
    #ax.set_title('Drug Release Over Time', fontsize=14)
    #st.pyplot(fig)

    # T 1/2 (Half-life) vs AUC (Area Under the Curve)
    st.write("Half-Life (T 1/2) vs Area Under the Curve (AUC)")
    fig, ax = plt.subplots()
    sns.scatterplot(x="T 1/2 of drug release (hour)", y="AUC brain (µg.h/ml)", hue="Route", palette="coolwarm", data=data, ax=ax)
    ax.set_title('Half-Life vs AUC', fontsize=14)
    st.pyplot(fig)

# EDA Analysis
def eda_analysis(data):
    st.write("### Exploratory Data Analysis (EDA)")
    st.write("Correlation Matrix")
    
    # Select only numeric columns for the correlation matrix
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr = numeric_data.corr()
    
    # Plot the correlation matrix
    fig = px.imshow(corr, text_auto=True, title="Correlation Matrix", template="plotly_dark", color_continuous_scale="RdBu_r")
    st.plotly_chart(fig)
    
## Functions for Tool 18
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
@st.cache_data(show_spinner=False)
def load_data18():
    disease_url = "https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/DrugSwitch/disease.csv"
    drug_url = "https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/DrugSwitch/drug.csv"
    metabolite_url = "https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/DrugSwitch/metabolite.csv"
    mirna_url = "https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/DrugSwitch/mirna.csv"
    mrna_url = "https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/DrugSwitch/mrna.csv"
    protein_url = "https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/DrugSwitch/protein.csv"
    
    # Load CSVs
    try:
        disease = pd.read_csv(disease_url, usecols=['Name', 'MyID', 'KEGG'])
        drug = pd.read_csv(drug_url, usecols=['generic_name', 'Sequence', 'drugbank_accession_number', 'groups', 'ECFP', 'ecfp_cluster', 'MyID'])
        metabolite = pd.read_csv(metabolite_url, usecols=['Common Name', 'SMILES', 'MyID'])
        mirna = pd.read_csv(mirna_url, usecols=['Symbol', 'MyID'])
        mrna = pd.read_csv(mrna_url, usecols=['Symbol', 'MyID'])
        protein = pd.read_csv(protein_url, usecols=['GeneCards', 'Sequence', 'Accession', 'STRING', 'SeqLength', 'KEGG', 'MyID'])
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None, None

    return disease, drug, metabolite, mirna, mrna, protein

# Merge the data
@st.cache_data(show_spinner=False)
def merge_data(disease, drug, metabolite, mirna, mrna, protein):
    merged_data = pd.merge(disease, drug, on="MyID", how="outer")
    merged_data = pd.merge(merged_data, metabolite, on="MyID", how="outer")
    merged_data = pd.merge(merged_data, mirna, on="MyID", how="outer")
    merged_data = pd.merge(merged_data, mrna, on="MyID", how="outer")
    merged_data = pd.merge(merged_data, protein, on="MyID", how="outer")
    
    # Rename columns
    merged_data = merged_data.rename(columns={
        'Name': 'Disease',
        'KEGG_x': 'Disease Pathway',
        'generic_name': 'Drug Name',
        'drugbank_accession_number': 'DrugBank ID',
        'groups': 'Drug Status',
        'Common Name': 'Metabolite',
        'SMILES_x': 'Metabolite SMILES',
        'Symbol_x': 'miRNA',
        'Symbol_y': 'mRNA',
        'GeneCards': 'Gene Name',
        'Sequence_x': 'Drug SMILES',
        'Accession': 'Protein Name (UniProt ID)',
        'STRING': 'PPI',
        'SeqLength': 'Protein Length',
        'KEGG_y': 'Enrichment (KEGG)',
        'Sequence_y': 'Protein Sequence'
    })
    
    return merged_data

# Enhanced Interactive Knowledge Graph Plot with color-coded nodes
import networkx as nx
import plotly.graph_objects as go
import numpy as np

def plot_interactive_knowledge_graph18(G, node_colors, node_categories):
    pos = nx.circular_layout(G)  # Circular layout for clarity

    # Extract positions for edges
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_y.append(y0)
        edge_y.append(y1)

    # Extract positions for nodes
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]

    # Create node trace with color and opacity
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[node for node in G.nodes()],
        textposition='top center',
        marker=dict(
            size=30,  # Increased node size
            color=[node_colors[node] for node in G.nodes()],  # Color per node category
            opacity=0.8,  # Add transparency
            line=dict(width=2)
        )
    )

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color='rgba(70,70,70,0.5)'),  # Add transparency to edges
        hoverinfo='none',
        mode='lines'
    )

    # Create frames for animation
    num_frames = 10  # Number of frames for animations
    frames = []
    for i in range(1, num_frames + 1):
        node_trace_frame = go.Scatter(
            x=np.array(node_x) * (1 + 0.02 * i),  # Slightly move nodes for animation
            y=np.array(node_y) * (1 + 0.02 * i),
            mode='markers+text',
            text=[node for node in G.nodes()],
            textposition='top center',
            marker=dict(
                size=30,
                color=[node_colors[node] for node in G.nodes()],
                opacity=max(0.5, 0.8 - 0.02 * i),  # Smooth transparency transition
                line=dict(width=2)
            )
        )
        frames.append(go.Frame(data=[node_trace_frame, edge_trace]))

    # Build figure with animations
    fig = go.Figure(
        data=[edge_trace, node_trace],
        frames=frames
    )

    # Add color-coded legend for node categories
    color_legend = []
    for category, color in node_categories.items():
        color_legend.append(
            go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=15, color=color, opacity=0.8),
                legendgroup=category,
                showlegend=True,
                name=category
            )
        )

    fig.add_traces(color_legend)

    # Update layout for better aesthetics
    fig.update_layout(
        title="Interactive Knowledge Graph with Entity Categories",
        showlegend=True,
        hovermode='closest',
        updatemenus=[{
            'buttons': [
                {
                    'args': [None, {"frame": {"duration": 300, "redraw": True},
                                    "fromcurrent": True}],
                    'label': 'Play',
                    'method': 'animate'
                },
                {
                    'args': [[None], {"frame": {"duration": 0, "redraw": True},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}],
                    'label': 'Pause',
                    'method': 'animate'
                }
            ],
            'direction': 'left',
            'pad': {"r": 10, "t": 87},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }],
    )

    return fig

# Define node categories and colors
node_categories = {
    "Disease": "red",
    "Disease Pathway": "orange",
    "Drug Name": "green",
    "Metabolite": "blue",
    "miRNA": "purple",
    "mRNA": "cyan",
    "Gene Name": "yellow",
    "Protein Name (UniProt ID)": "pink",
    "PPI": "brown",
    "Enrichment (KEGG)": "lime"
}
# Function to build the knowledge graph
# Build Graph Function with color-coding for entities
def build_knowledge_graph18(df, node_categories):
    G = nx.Graph()
    node_colors = {}

    # Safeguard: Check for column existence before adding nodes and edges
    for _, row in df.iterrows():
        # Disease
        if 'Disease' in df.columns and not pd.isna(row['Disease']):
            G.add_node(row['Disease'])
            node_colors[row['Disease']] = node_categories["Disease"]
        
        # Drug Name
        if 'Drug Name' in df.columns and not pd.isna(row['Drug Name']):
            G.add_node(row['Drug Name'])
            node_colors[row['Drug Name']] = node_categories["Drug Name"]
        
        # Metabolite
        if 'Metabolite' in df.columns and not pd.isna(row['Metabolite']):
            G.add_node(row['Metabolite'])
            node_colors[row['Metabolite']] = node_categories["Metabolite"]
            G.add_edge(row['Drug Name'], row['Metabolite'])

        # miRNA
        if 'miRNA' in df.columns and not pd.isna(row['miRNA']):
            G.add_node(row['miRNA'])
            node_colors[row['miRNA']] = node_categories["miRNA"]
            G.add_edge(row['Metabolite'], row['miRNA'])

        # mRNA
        if 'mRNA' in df.columns and not pd.isna(row['mRNA']):
            G.add_node(row['mRNA'])
            node_colors[row['mRNA']] = node_categories["mRNA"]
            G.add_edge(row['miRNA'], row['mRNA'])

        # Gene Name
        if 'Gene Name' in df.columns and not pd.isna(row['Gene Name']):
            G.add_node(row['Gene Name'])
            node_colors[row['Gene Name']] = node_categories["Gene Name"]
            G.add_edge(row['mRNA'], row['Gene Name'])

        # Protein Name (UniProt ID)
        if 'Protein Name (UniProt ID)' in df.columns and not pd.isna(row['Protein Name (UniProt ID)']):
            G.add_node(row['Protein Name (UniProt ID)'])
            node_colors[row['Protein Name (UniProt ID)']] = node_categories["Protein Name (UniProt ID)"]
            G.add_edge(row['Gene Name'], row['Protein Name (UniProt ID)'])

    return G, node_colors


# Function to perform analysis and plotting
# Function to perform analysis and plotting
def perform_analysis(merged_data):
    st.markdown("### Additional Analysis")

    # Example: Distribution of Drug Status
    st.markdown("#### Distribution of Drug Status")
    drug_status_count = merged_data['Drug Status'].value_counts()
    st.bar_chart(drug_status_count)

    # Example: Distribution of Protein Length (Using Plotly Express for Streamlit)
    st.markdown("#### Distribution of Protein Length")
    fig = px.histogram(merged_data, x='Protein Length', nbins=20, title='Distribution of Protein Length')
    st.plotly_chart(fig)

#Functions for Tool 19:
# Load the data from URLs
@st.cache_data
def load_data19():
    # Load combine_score.csv
    combine_score = pd.read_csv(
        "https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/DrugEnricher/DrugEnricher.csv", 
        sep=",",  
        encoding='ISO-8859-1',
        on_bad_lines='skip'
    )
    
    # Load Path2gene_wikipathway.txt
    path2gene_wikipathway = pd.read_csv(
        "https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/DrugEnricher/Path2gene_wikipathway.txt", 
        sep="\t", 
        encoding='ISO-8859-1',
        on_bad_lines='skip'
    )
    
    # Clean the 'Genes' column to handle gene lists correctly
    path2gene_wikipathway['Genes'] = path2gene_wikipathway['Genes'].str.replace('&#10;', ',')
    
    return combine_score, path2gene_wikipathway

#Functions for Tool 20:
# Load the data
@st.cache_data
def load_data20():
    try:
        # Load the data from the provided URL
        biomarker_data = pd.read_excel("https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/DrugEnricher/Pharmacogenomic_Biomarkers_FDA.xlsx", sheet_name=0)

        # Correctly rename the columns based on the actual data
        biomarker_data.columns = ['Drug', 'TherapeuticArea', 'Biomarker', 'LabelingSections']
        
        return biomarker_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error



# Streamlit App
#st.title("Screengenix: Comprehensive Molecular Analysis")
# Sidebar for tool selection
#st.sidebar.header("Screengenix Tools")
#tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
#    "Tool 1: Ro5 Analysis", 
#    "Tool 2: Descriptor Analysis", 
#    "Tool 3: Similarity Analysis", 
#    "Tool 4: Compound Clustering", 
#    "Tool 5: Molecule Insight", 
#    "Screengenix-ML", 
#    "Screengenix-DL"
#])
# Streamlit App (Second option)
#st.title("Screengenix: Comprehensive Molecular Analysis")

# Sidebar for tool selection
#st.sidebar.header("Home: Screengenix Tools")
#selected_tool = st.sidebar.selectbox(
#    "Choose a tool",
#    [
#        "Tool 1: Ro5 Analysis",
#        "Tool 2: Descriptor Analysis",
#        "Tool 3: Similarity Analysis",
#        "Tool 4: Compound Clustering",
#        "Tool 5: Molecule Insight",
#        "Tool 6: Screengenix-ML",
#        "Tool 7: Screengenix-DL"
#    ]
#)

# Start the pinging function in a background thread
threading.Thread(target=ping_app, daemon=True).start()

#Never Sleep
import streamlit as st
import requests
import threading

# UptimeRobot API key for chemgenix.streamlit.app
UPTIMEROBOT_API_KEY = 'm797700399-510619070bf44b75776b871c'

# UptimeRobot API URL to get monitors
UPTIMEROBOT_API_URL = 'https://api.uptimerobot.com/v2/getMonitors'

# Function to get monitor ID
def get_monitor_id():
    payload = {
        'api_key': UPTIMEROBOT_API_KEY,
        'format': 'json'
    }
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    
    # Make POST request to UptimeRobot API
    response = requests.post(UPTIMEROBOT_API_URL, data=payload, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        monitors = data.get('monitors', [])
        
        # Loop through monitors to find chemgenix.streamlit.app
        for monitor in monitors:
            if monitor['url'] == 'https://chemgenix.streamlit.app/':
                st.session_state.monitor_id = monitor['id']
                print(f"Name: {monitor['friendly_name']}, ID: {monitor['id']}, URL: {monitor['url']}")
                return
    else:
        st.session_state.monitor_id = None
        print('Failed to retrieve monitors. Status code:', response.status_code)

# Asynchronous function to prevent blocking Streamlit
def fetch_monitor_in_background():
    if 'monitor_id' not in st.session_state:
        # Start the API call in a separate thread
        thread = threading.Thread(target=get_monitor_id)
        thread.start()

#END

#CSS

import streamlit as st
import streamlit.components.v1 as components

# Title of the App
# Fetch and display the logo
response = requests.get("https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/SreenGenix_Logo.jpg")
img = Image.open(BytesIO(response.content))

# Title of the App with custom styling
st.markdown(
    """
    <style>
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        body {
            background-color: #ecf0f1;
        }
        .main-title {
            color: #3498db;
            font-size: 50px;
            font-weight: bold;
            margin-bottom: 0;
            animation: fadeIn 2s ease-out;
            background-color: #3498db;
            color: white;
            padding: 10px;
            border-radius: 10px;
        }
        .subtitle {
            color: #2ecc71;
            font-size: 30px;
            font-weight: lighter;
            margin-top: 0;
            animation: fadeIn 2s ease-out 1s;
            background-color: #f9f9f9;
            padding: 5px;
            border-radius: 5px;
        }
        .sidebar-title {
            color: #e74c3c;
            font-size: 30px;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .sidebar-menu:hover {
            color: #3498db;
            transition: color 0.3s ease;
        }
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #f1f1f1;
            color: #333;
            text-align: center;
            padding: 10px;
            font-size: 12px;
            border-top: 1px solid #ddd;
            animation: fadeIn 2s ease-out 2s;
        }
        .footer h4 {
            margin: 0;
            color: #3498db;
        }
        .footer p {
            margin: 0;
            color: #555;
        }
    </style>
    <div class='main-title' style='font-size: 30px; text-align: center;'>
        PhytoChemX
        <img src="https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/SreenGenix_Logo.jpg" alt="Logo" style="height: 40px; margin-left: 10px; vertical-align: middle;">
    </div>
    """,
    unsafe_allow_html=True
)

import streamlit as st
import time

# Sidebar with advanced menu options
with st.sidebar:
    selected_tool = option_menu(
        "PhytoChemX",
        [
            "Home",
            "Tool 1: DataExtract",
            "Tool 2: Ro5Scan",
            "Tool 3: Descripto",
            "Tool 4: SimAnalyzr",
            "Tool 5: ChemCluster",
            "Tool 6: MolInsight",
            "Tool 7: ScreenGen-ML",
            "Tool 8: ScreenGen-DL",
            "Tool 9: TargetPredictor",
            "Tool 10: SmartTox",
            "Tool 11: DDI-Predictor",
            "Tool 12: CellDrugFinder",
            "Tool 13: NetPharm",
            "Tool 14: QSARGx",
            "Tool 15: ChemSyn",
            "Tool 16: ChemPROTAC",
            "Tool 17: NanoMedScore",
            "Tool 18: DrugSwitch",
            "Tool 19: DrugEnricher",
            "Tool 20: DrugMarker",
            "Team",
            "About"
        ],
        icons=[
            "house", "gear", "graph-up", "calculator", "lightbulb", "puzzle", "cloud", "robot", "activity", "filter", "shield", "capsule-pill", "check-square", "globe", "compass", "fire", "search", "power", "unlock", "tag", "paperclip", "people", "info-circle"
        ],
        menu_icon="cast", 
        default_index=0, 
        orientation="vertical",
        styles={
            "container": {"padding": "0!important", "background-color": "rgba(255, 255, 255, 0.5)"},
            "icon": {"color": "#e74c3c", "font-size": "25px"},
            "nav-link": {"font-size": "18px", "text-align": "left", "margin": "0px", "--hover-color": "#eaeaea"},
            "nav-link-selected": {"background-color": "#3498db"},
        }
    )

# Initialize session state variables if they don't exist
for key, default in [
    ('selected_tool', 'Home')
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Main content based on selection
if selected_tool == "Home":
    st.markdown("<h4 style='color: #3498db; text-align: center;'>Welcome to PhytoChemX!</h4>", unsafe_allow_html=True)
    
    # Add the header image with styling
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <img src="https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/Images/Header.jpg" 
             style="width: 95%; max-width: 1000px; height: auto; border-radius: 15px; 
                    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);" 
             alt="PhytoChemX Image"/>
    </div>
    """, unsafe_allow_html=True)

    # First Section: CSIR-CIMAP Description with Image Rotation
    st.markdown("<h5 style='text-align: center;'>CSIR-Central Institute of Medicinal and Aromatic Plants (CSIR-CIMAP)</h5>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        <div style="padding: 20px; font-size: 16px; line-height: 1.5; text-align: justify;">
            <p><b>CSIR-Central Institute of Medicinal and Aromatic Plants (CSIR-CIMAP)</b> is a multidisciplinary research institute of CSIR, India, focusing on the potential of medicinal and aromatic plants (MAPs) through cultivation, chemical characterization, extraction, and formulation of bioactive phytomolecules. CSIR-CIMAP has played a key role in positioning India as a leader in the global production of mints, vetiver, and other aromatic plants.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Set up the image display block with a dynamic element
        image_placeholder = st.empty()

        # Image URLs for rotation
        image_urls = [
            "https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/Aroma/DamaskRose.jpg",
            "https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/Aroma/jammu_monarda.jpg",
            "https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/Aroma/Khus3.jpg",
            "https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/Aroma/Lavender1.jpg",
            "https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/Aroma/Lemongrass5.jpg",
            "https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/Aroma/Menthol_mint_1.jpg",
            "https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/Aroma/Muskbala2.jpg",
            "https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/Aroma/Palmarosa_close_view2.jpg",
            "https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/Aroma/Patchouli2.jpg",
            "https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/Aroma/Rosagrass_Cultivation_temp.jpg",
            "https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/Aroma/Rosemary.jpg",
            "https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/Aroma/SeaWormwood.jpg",
            "https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/Aroma/Tagetes6.jpg",
            "https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/Aroma/Thimsingli2.jpg",
            "https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/Aroma/Tulsi4.jpg"
          
        ]
        
        # Rotate images a set number of times
        total_cycles = 1  # Set to 1 cycles (can be adjusted)
        image_count = len(image_urls)
        
        for cycle in range(total_cycles):  # Limited cycles
            for i in range(image_count):
                with image_placeholder:
                    st.image(image_urls[i], width=500)  # Adjust the width for the desired size
                time.sleep(1)  # Sleep for 1 second before changing the image

    # Add the 6 image sections with clickable links and headings
    col1, col2 = st.columns(2)

    # Create columns for layout with equal width (1:1 ratio)
    col1, col2 = st.columns([1, 1])  # Equal width for text and images
    
    with col1:
        st.markdown("**PhytoChemX?**")
        st.markdown(
            """
            <div style="text-align: center; border: 1px solid #ddd; padding: 15px; border-radius: 10px; margin-bottom: 20px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); transition: transform 0.3s;">
                <img src="https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/Whatsitallabout.png" width="300">
                <p style="text-align: justify; margin-top: 10px;">An advanced platform to explore phytochemicals, analyze their properties, and study medicinal potential.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
        st.markdown("**Modules**")
        st.markdown(
            """
            <div style="text-align: center; border: 1px solid #ddd; padding: 15px; border-radius: 10px; margin-bottom: 20px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); transition: transform 0.3s;">
                <img src="https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/PackagesTools.png" width="300">
                <p style="text-align: justify; margin-top: 10px;">Offers AI-powered tools for scaffold analysis, and QSAR modeling in drug discovery.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
        st.markdown("**Key Research Questions**")
        st.markdown(
            """
            <div style="text-align: center; border: 1px solid #ddd; padding: 15px; border-radius: 10px; margin-bottom: 20px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); transition: transform 0.3s;">
                <img src="https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/ResearchQuestions.png" width="300">
                <p style="text-align: justify; margin-top: 10px;">Investigates interactions of plant-derived molecules and evaluates pharmacokinetics.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
        st.markdown("**Explore related Publications**")
        st.markdown(
            """
            <div style="text-align: center; border: 1px solid #ddd; padding: 15px; border-radius: 10px; margin-bottom: 20px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); transition: transform 0.3s;">
                <img src="https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/PublicationsResources.png" width="300">
                <p style="text-align: justify; margin-top: 10px;">Browse curated studies, peer-reviewed articles, and scientific databases on phytochemical research.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col2:
        st.markdown("**Getting started with PhytoChemX**")
        st.markdown(
            """
            <div style="text-align: center; border: 1px solid #ddd; padding: 15px; border-radius: 10px; margin-bottom: 20px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); transition: transform 0.3s;">
                <img src="https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/GetgoingwithPhytoChemX.png" width="300">
                <p style="text-align: justify; margin-top: 10px;">Designed for all levels, our user-friendly interface and tutorials streamline research processes.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
        st.markdown("**Meet Team behind PhytoChemX**")
        st.markdown(
            """
            <div style="text-align: center; border: 1px solid #ddd; padding: 15px; border-radius: 10px; margin-bottom: 20px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); transition: transform 0.3s;">
                <img src="https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/Team.png" width="300">
                <p style="text-align: justify; margin-top: 10px;">A team of experts in bioinformatics, chemistry, and AI collaborates to enhance PhytoChemX.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
        st.markdown("**Our collaborations**")
        st.markdown(
            """
            <div style="text-align: center; border: 1px solid #ddd; padding: 15px; border-radius: 10px; margin-bottom: 20px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); transition: transform 0.3s;">
                <img src="https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/Collaborations.png" width="300">
                <p style="text-align: justify; margin-top: 10px;">Partnering with universities, biotech firms, and pharma companies to drive scientific progress.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
        st.markdown("**The impact of PhytoChemX**")
        st.markdown(
            """
            <div style="text-align: center; border: 1px solid #ddd; padding: 15px; border-radius: 10px; margin-bottom: 20px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); transition: transform 0.3s;">
                <img src="https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/Impact.png" width="300">
                <p style="text-align: justify; margin-top: 10px;">Advancing drug discovery through AI-driven compound prediction and ADMET analysis models.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Footer section
    st.markdown("""
    <div style="padding: 20px; text-align: center; color: #555;">
        <p>PhytoChemX | A Platform for Medicinal and Aromatic Plants Research</p>
    </div>
    """, unsafe_allow_html=True)

elif selected_tool == "Tool 1: DataExtract":
    st.markdown("<h4 style='color: #3498db;'>Tool 1: ChEMBL Target Profiling</h4>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 18px;'>Data Extraction & ChEMBL Drug-Target Profiling.</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Initialize session state variables for ChEMBL
    if 'query_protein_chembl' not in st.session_state:
        st.session_state.query_protein_chembl = ""
    
    if 'query_chembl_id' not in st.session_state:
        st.session_state.query_chembl_id = ""
    
    st.text_input("Enter Target", key="query_protein_chembl")
    
    if st.button("Submit", key="submit_chembl_protein") and st.session_state.query_protein_chembl:
        with st.spinner("Fetching Data..."):
            molecules = Protein_data_extractor(st.session_state.query_protein_chembl)  # Placeholder function
            st.write("Data Loaded from ChEMBL", molecules)
            st.session_state.molecules_chembl = molecules
    
    st.text_input("Enter Chembl id", key="query_chembl_id")
    
    if st.button("Submit", key="submit_chembl_id") and st.session_state.query_chembl_id:
        with st.spinner("Fetching Data..."):
            api_url = "https://www.ebi.ac.uk/chembl/api/data/activity"
            headers = {"Content-Type": "application/json"}
            params = {
                'target_chembl_id': st.session_state.query_chembl_id,
                'assay_type': 'B',
                'type': 'IC50',
                'format': 'json',
                'relation': '=',
                'standard_units': 'nM',
                'limit': '1000'
            }
            response = requests.get(api_url, params=params)
            data = response.json()
            bioactivities = data['activities']
    
            num_page = math.floor(data['page_meta']['total_count'] / 1000)
            for i in range(num_page):
                response = requests.get("https://www.ebi.ac.uk" + data['page_meta']['next'])
                data = response.json()
                bioactivities.extend(data['activities'])
    
            molecule_chembl_id = []
            ic50 = []
            units = []
            smiles = []
    
            for activity in bioactivities:
                molecule_chembl_id.append(activity['molecule_chembl_id'])
                ic50.append(activity['standard_value'])
                units.append(activity['standard_units'])
                smiles.append(activity['canonical_smiles'])
    
            st.session_state.molecules_chembl = pd.DataFrame(
                {"molecule_chembl_id": molecule_chembl_id, "ic50": ic50, "units": units, "smiles": smiles})
            st.write(st.session_state.molecules_chembl)  # <--- FIXED HERE
            st.success("Data extraction is complete")
    
    if st.session_state.get('molecules_chembl') is not None:  # Check if molecules_chembl exists
        query_smiles_chembl = st.text_input("Enter query molecule SMILES for similarity search (CHEMBL)", key="chembl_smiles")
        submit_button_chembl_smiles = st.button("Submit", key="submit_chembl_smiles")
    
        if submit_button_chembl_smiles and query_smiles_chembl:
            # Set the similarity metric to "Tanimoto"
            selected_metric = "Tanimoto"  # No need to ask user for metric, it's always Tanimoto
    
            # Continue with the molecule query and similarity calculation
            query_mol = Chem.MolFromSmiles(query_smiles_chembl)
            if query_mol is None:
                st.error("Invalid SMILES string. Please check the format.")
            else:
                query_fp = MACCSkeys.GenMACCSKeys(query_mol)
                valid_fps = [MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smiles)) for smiles in st.session_state.molecules_chembl["smiles"] if Chem.MolFromSmiles(smiles) is not None]
    
                if not valid_fps:
                    st.error("No valid fingerprints found. Check the SMILES in the data.")
                else:
                    # Calculate similarity using the Tanimoto metric
                    similarities = calculate_similarity(query_fp, valid_fps, selected_metric)
                    st.session_state.molecules_chembl["Similarity"] = similarities
                    st.session_state.similarity_results_chembl = st.session_state.molecules_chembl.sort_values("Similarity", ascending=False)
    
                    st.write("Similarity Results", st.session_state.similarity_results_chembl)  # <--- FIXED HERE
                    plot_similarity_results(st.session_state.similarity_results_chembl, "molecule_chembl_id")
    
                    csv = convert_df(st.session_state.similarity_results_chembl)
                    st.download_button(
                        label="Download data as CSV",
                        data=csv,
                        file_name=f'Compounds_data_{st.session_state.query_chembl_id}_CHEMBL.csv',
                        mime='text/csv',
                    )
    else:
        st.error("Please load data from ChEMBL first.")

     

#with tab2:
#    st.header("Tool 2: Ro5 Analysis")
#if tool_selection == "Tool 2: Ro5 Analysis":
#    st.header("Tool 2: Ro5 Analysis")
# Tool 2: Ro5 Analysis
elif selected_tool == "Tool 2: Ro5Scan":
    st.markdown("<h4 style='color: #3498db;'>Tool 2: Ro5 Analysis</h4>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 18px;'>Analyze the compliance of compounds with Lipinski's Ro5 to assess their drug-likeness.</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Uploading the CSV file for analysis
    uploaded_file_tool1 = st.file_uploader("Upload CSV file with SMILES data for Ro5 Analysis", type="csv", key="tool1")
    
    if uploaded_file_tool1:
        molecules_tool1 = pd.read_csv(uploaded_file_tool1)
        st.write("Data Preview", molecules_tool1.head())

        # Standardize column names to lowercase
        molecules_tool1.columns = [col.lower() for col in molecules_tool1.columns]
        
        # Check for Ro5 calculated columns
        ro5_columns = {"molecular_weight", "n_hba", "n_hbd", "logp", "ro5_fulfilled"}
        existing_ro5_columns = set(molecules_tool1.columns).intersection(ro5_columns)

        if existing_ro5_columns == ro5_columns:
            # Case 1: File already contains Ro5-calculated columns, return the file as-is
            st.write("Ro5 properties are already calculated in the uploaded file.")
            st.write("Data with Ro5 Properties", molecules_tool1)

            # Ro5 Fulfilled and Violated subsets
            molecules_ro5_fulfilled = molecules_tool1[molecules_tool1["ro5_fulfilled"]]
            molecules_ro5_violated = molecules_tool1[~molecules_tool1["ro5_fulfilled"]]
        
        else:
            # Case 2: Only SMILES column is available, recalculate Ro5 properties
            smiles_column = next((col for col in molecules_tool1.columns if 'smile' in col), None)
            
            if smiles_column:
                ro5_properties = molecules_tool1[smiles_column].apply(calculate_ro5_properties)
                molecules_tool1 = pd.concat([molecules_tool1, ro5_properties], axis=1)

                # Convert invalid data types (e.g., None) to NaN for safe display
                molecules_tool1 = molecules_tool1.fillna(np.nan)
                st.write("Data with Ro5 Properties", molecules_tool1)

                # Ro5 Fulfilled and Violated subsets
                molecules_ro5_fulfilled = molecules_tool1[molecules_tool1["ro5_fulfilled"]]
                molecules_ro5_violated = molecules_tool1[~molecules_tool1["ro5_fulfilled"]]
            else:
                # Display an error message if there's no SMILES column
                st.error("The uploaded file must contain a 'smiles' column.")
                st.stop()  # Stop the execution of the Streamlit script

        # Preparing downloadable results
        result_csv_tool1 = io.BytesIO()
        molecules_tool1.to_csv(result_csv_tool1, index=False)
        result_csv_tool1.seek(0)
        st.download_button("Download Results CSV", result_csv_tool1, "results.csv", "text/csv")

        # Download buttons for fulfilled and violated subsets
        fulfilled_csv_tool1 = io.BytesIO()
        molecules_ro5_fulfilled.to_csv(fulfilled_csv_tool1, index=False)
        fulfilled_csv_tool1.seek(0)
        st.download_button("Download Ro5 Fulfilled Data CSV", fulfilled_csv_tool1, "ro5_fulfilled.csv", "text/csv")

        violated_csv_tool1 = io.BytesIO()
        molecules_ro5_violated.to_csv(violated_csv_tool1, index=False)
        violated_csv_tool1.seek(0)
        st.download_button("Download Ro5 Violated Data CSV", violated_csv_tool1, "text/csv")

        # Calculating statistics for fulfilled and violated molecules
        if not molecules_ro5_fulfilled.empty:
            molecules_ro5_fulfilled_stats = calculate_mean_std(molecules_ro5_fulfilled[["molecular_weight", "n_hba", "n_hbd", "logp"]])
            thresholds = {"molecular_weight": 500, "n_hba": 10, "n_hbd": 5, "logp": 5}
            scaled_threshold = 5
            properties_labels = ["Molecular weight", "HBA", "HBD", "LogP"]

            # Display Radar Plot for fulfilled
            st.subheader("Radar Plot: Ro5 Fulfilled")
            fig_radar_fulfilled = io.BytesIO()
            plot_radar(molecules_ro5_fulfilled_stats, thresholds, scaled_threshold, properties_labels, output_path=fig_radar_fulfilled)
            fig_radar_fulfilled.seek(0)
            st.download_button("Download Ro5 Fulfilled Radar Plot", fig_radar_fulfilled, "ro5_fulfilled_radar.png", "image/png")

        if not molecules_ro5_violated.empty:
            molecules_ro5_violated_stats = calculate_mean_std(molecules_ro5_violated[["molecular_weight", "n_hba", "n_hbd", "logp"]])

            # Display Radar Plot for violated
            st.subheader("Radar Plot: Ro5 Violated")
            fig_radar_violated = io.BytesIO()
            plot_radar(molecules_ro5_violated_stats, thresholds, scaled_threshold, properties_labels, output_path=fig_radar_violated)
            fig_radar_violated.seek(0)
            st.download_button("Download Ro5 Violated Radar Plot", fig_radar_violated, "ro5_violated_radar.png", "image/png")

    else:
        st.error("Please upload a CSV file to proceed.")



#with tab3:
#    st.header("Tool 3: Descriptor Analysis")
#elif tool_selection == "Tool 3: Descriptor Analysis":
#    st.header("Tool 3: Descriptor Analysis")
# Tool 3: Descriptor Analysis
# Tool 3: Descriptor Analysis
elif selected_tool == "Tool 3: Descripto":
    st.markdown("<h4 style='color: #3498db;'>Tool 3: Descriptor Analysis</h4>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 18px;'>Calculate molecular descriptors and visualize correlations between them.</p>", unsafe_allow_html=True)
    st.markdown("---")
    uploaded_file_tool2 = st.file_uploader("Upload CSV file with SMILES data for Descriptor Analysis", type="csv", key="tool2")
    if uploaded_file_tool2:
        data_tool2 = process_file(uploaded_file_tool2)

        if not data_tool2.empty:
            st.write("Calculated Descriptors", data_tool2)

            result_csv_tool2 = io.BytesIO()
            data_tool2.to_csv(result_csv_tool2, index=False)
            result_csv_tool2.seek(0)
            st.download_button("Download Descriptors CSV", result_csv_tool2, "descriptors.csv", "text/csv")

            # Correlation plot
            st.subheader("Correlation Plot")
            
            # Ensure correlation is calculated on numeric data only
            numeric_data_tool2 = data_tool2.select_dtypes(include=[np.number]).dropna()
            if numeric_data_tool2.shape[1] > 1:
                correlation = numeric_data_tool2.corr()
                fig, ax = plt.subplots()
                sns.heatmap(correlation, annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)
            else:
                st.error("Not enough numeric data for correlation plot.")

            # Scatter plot
            st.subheader("Scatter Plot")
            x_axis = st.selectbox("Select X-axis descriptor", options=numeric_data_tool2.columns)
            y_axis = st.selectbox("Select Y-axis descriptor", options=numeric_data_tool2.columns)
            scatter_fig = px.scatter(numeric_data_tool2, x=x_axis, y=y_axis, labels={x_axis: x_axis, y_axis: y_axis})
            st.plotly_chart(scatter_fig)
        else:
            st.error("The uploaded file does not contain the required data.")

#with tab4:
#    st.header("Tool 4: Similarity Analysis")
#elif tool_selection == "Tool 4: Similarity Analysis":
#    st.header("Similarity Analysis")
# Tool 4: Similarity Analysis
elif selected_tool == "Tool 4: SimAnalyzr":
    st.markdown("<h4 style='color: #3498db;'>Tool 4: Similarity Analysis</h4>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 18px;'>Compare molecular fingerprints and visualize the Tanimoto similarity between molecules.</p>", unsafe_allow_html=True)
    st.markdown("---")

    # File uploader for CSV with SMILES data
    uploaded_file_tool3 = st.file_uploader("Upload CSV file with SMILES data for Similarity Analysis", type="csv", key="tool3")

    # Dropdown to select the similarity metric
    similarity_metric_tool3 = st.selectbox("Select Similarity Metric", 
                                           ["Tanimoto", "Dice", "Cosine", "Sokal", "Russel", 
                                            "RogotGoldberg", "Kulczynski", "McConnaughey", 
                                            "Tversky", "BraunBlanquet", "Morgan"], key="similarity_metric")

    # Input for query molecule in SMILES format
    query_mol_tool3 = st.text_input("Enter query molecule in SMILES format", key="query_mol")

    # Checkbox states
    st.session_state.show_heatmap = st.checkbox("Show Similarity Heatmap", value=st.session_state.get("show_heatmap", False))
    st.session_state.show_dendrogram = st.checkbox("Show Dendrogram", value=st.session_state.get("show_dendrogram", False))
    st.session_state.show_distribution = st.checkbox("Show Similarity Distribution", value=st.session_state.get("show_distribution", False))
    st.session_state.show_enrichment = st.checkbox("Show Enrichment Plot", value=st.session_state.get("show_enrichment", False))
    
    # Separate checkboxes for PCA and t-SNE
    st.session_state.show_pca = st.checkbox("Show PCA Plot", value=st.session_state.get("show_pca", False))
    st.session_state.show_tsne = st.checkbox("Show t-SNE Plot", value=st.session_state.get("show_tsne", False))

    # Submit button for the query input
    submit_button = st.button("Submit", key="submit")

    # Check if the submit button was clicked
    if submit_button:
        if uploaded_file_tool3:
            data_tool3 = load_data(uploaded_file_tool3)

            # Ensure the 'smiles' column exists (case insensitive)
            if any(col.lower() == 'smiles' for col in data_tool3.columns):
                smiles_col = [col for col in data_tool3.columns if col.lower() == 'smiles'][0]

                if query_mol_tool3:
                    # Generate MACCS keys fingerprint for the query molecule
                    query_fp_tool3 = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(query_mol_tool3))

                    # Generate fingerprints for each molecule in the dataset
                    fps_tool3 = [MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smiles)) for smiles in data_tool3[smiles_col]]

                    # Calculate similarity using the selected metric
                    similarity_tool3 = calculate_similarity(query_fp_tool3, fps_tool3, similarity_metric_tool3)

                    # Add similarity results to the DataFrame
                    data_tool3["Similarity"] = similarity_tool3

                    # Display the similarity results, sorted by similarity
                    st.write("Similarity Results", data_tool3.sort_values("Similarity", ascending=False))

                    # Call the plotting function
                    display_plots(fps_tool3, similarity_tool3)

                    # Handle CSV download functionality for the results
                    result_csv_tool3 = io.BytesIO()
                    data_tool3.to_csv(result_csv_tool3, index=False)
                    result_csv_tool3.seek(0)
                    st.download_button("Download Similarity Results CSV", result_csv_tool3, "similarity_results.csv", "text/csv")
                else:
                    st.error("Please enter a query molecule in SMILES format.")
            else:
                st.error("The uploaded file must contain a 'smiles' column.")


#with tab5:
#    st.header("Tool 5: Compound Clustering")
#elif tool_selection == "Tool 5: Compound Clustering":
#    st.header("Tool 5: Compound Clustering")

# Tool 5: Compound Clustering
elif selected_tool == "Tool 5: ChemCluster":
    st.markdown("<h4 style='color: #3498db;'>Tool 5: Compound Clustering</h4>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 18px;'>Cluster compounds based on molecular fingerprints.</p>", unsafe_allow_html=True)
    st.markdown("---")

    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV file containing SMILES data:", type=["csv"])

    if uploaded_file is not None:
        df = load_molecules5(uploaded_file)
        if not df.empty:
            st.write("Data Preview:", df.head())

            # Create molecules and generate fingerprints
            compounds = create_molecules5(df)
            fingerprints = generate_fingerprints5(compounds)

            # Calculate the similarity matrix
            num_fps = len(fingerprints)
            similarity_matrix = np.zeros((num_fps, num_fps))
            for i in range(num_fps):
                for j in range(num_fps):
                    similarity_matrix[i, j] = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])

            # Tanimoto similarity and distance matrix
            st.header("Tanimoto Similarity")
            if st.button("Compute Tanimoto Similarity"):
                if len(fingerprints) > 1:
                    sim = DataStructs.TanimotoSimilarity(fingerprints[0], fingerprints[1])
                    st.write(f"Tanimoto similarity: {sim:.2f}, distance: {1-sim:.2f}")
                    st.write("Distance Matrix (first 5 elements):", tanimoto_distance_matrix5(fingerprints)[:5])
                else:
                    st.warning("Not enough fingerprints to compute similarity.")

            # Clustering
            st.header("Clustering")
            cutoff = st.slider(
                "Clustering Cutoff",
                min_value=0.1,
                max_value=1.0,
                value=0.3,
                step=0.01
            )
            clusters = cluster_fingerprints5(fingerprints, cutoff)

            # Add checkboxes for plot visibility
            show_cluster_size_dist = st.checkbox("Show Cluster Size Distribution", value=False)
            show_similarity_heatmap = st.checkbox("Show Similarity Heatmap", value=False)
            show_pca = st.checkbox("Show PCA Plot", value=False)
            show_dendrogram = st.checkbox("Show Dendrogram", value=False)

            if show_cluster_size_dist:
                st.write("Cluster Size Distribution")
                fig, ax = plt.subplots(figsize=(15, 4))
                ax.set_xlabel("Cluster index")
                ax.set_ylabel("Number of molecules")
                ax.bar(range(1, len(clusters) + 1), [len(c) for c in clusters], color='darkblue', edgecolor='black', lw=1)
                st.pyplot(fig)

            # Cluster analysis
            num_clusters = len(clusters)
            largest_cluster_size = len(clusters[0]) if clusters else 0
            st.write(f"Number of clusters: {num_clusters}")
            st.write(f"Number of molecules in largest cluster: {largest_cluster_size}")
            
            # Ensure clusters have enough elements for similarity calculations
            if num_clusters > 1 and len(clusters[0]) > 1 and len(clusters[1]) > 0:
                sim_same_cluster = DataStructs.TanimotoSimilarity(fingerprints[clusters[0][0]], fingerprints[clusters[0][1]])
                sim_diff_cluster = DataStructs.TanimotoSimilarity(fingerprints[clusters[0][0]], fingerprints[clusters[1][0]])
                st.write(f"Similarity between two random points in the same cluster: {sim_same_cluster:.2f}")
                st.write(f"Similarity between two random points in different clusters: {sim_diff_cluster:.2f}")
            else:
                st.warning("Not enough molecules in clusters to compute similarity.")

            # Visualize molecules from clusters (basic visualization)
            st.header("Cluster Visualization")
            def visualize_clusters(clusters, compounds, num_examples=10):
                for i, cluster in enumerate(clusters[:10]):  # Limit to first 10 clusters
                    if len(cluster) > 0:
                        st.write(f"Cluster {i + 1} - {len(cluster)} molecules")
                        # Display the first few SMILES strings as a placeholder for actual molecule images
                        for idx in cluster[:num_examples]:
                            st.write(compounds[idx][1])  # Displaying the chembl_id as placeholder

            visualize_clusters(clusters, compounds)
            
            if similarity_matrix is not None and similarity_matrix.size > 0:
                if show_similarity_heatmap:
                    plot_similarity_heatmap5(similarity_matrix)
            else:
                st.warning("Similarity matrix is empty or not computed correctly.")

            if show_pca:
                plot_pca5(fingerprints, clusters)

            if show_dendrogram:
                plot_dendrogram5(fingerprints)
            
            # Save molecules to SDF
            if st.button("Save Molecules from Largest Cluster"):
                try:
                    # Initialize in-memory file
                    sdf_file = StringIO()  # Use StringIO for text mode file operations
                    
                    # Initialize SDF writer
                    sdf_writer = Chem.SDWriter(sdf_file)
                    
                    # Write molecules to the SDF file
                    for index in clusters[0]:
                        mol, label = compounds[index]
                        if mol is not None:
                            mol.SetProp("_Name", label)
                            sdf_writer.write(mol)
                        else:
                            st.warning(f"Invalid molecule at index {index}. Skipping.")
                    
                    # Close the SDF writer
                    sdf_writer.close()
            
                    # Save the in-memory file to the app's temporary directory
                    sdf_file.seek(0)
                    st.download_button(
                        label="Download Molecules as SDF",
                        data=sdf_file.getvalue(),
                        file_name="molecule_set_largest_cluster.sdf",
                        mime="chemical/x-sdfile"
                    )
                except Exception as e:
                    st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload a CSV file to proceed.")


#with tab6:
#    st.header("Tool 6: Molecule Insight")
#elif tool_selection == "Tool 6: Molecule Insight":
#    st.header("Tool 6: Molecule Insight")
# Tool 6: Molecule Insight
elif selected_tool == "Tool 6: MolInsight":
    st.markdown("<h4 style='color: #3498db;'>Tool 6: Molecule Insight</h4>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 18px;'>FMCS algorithm that's used to find the maximum common substructure</p>", unsafe_allow_html=True)
    st.markdown("---")
    uploaded_file_tool5 = st.file_uploader("Upload a CSV or SDF file", type=["csv", "sdf"])

    if uploaded_file_tool5 is not None:
        # Load molecules
        molecules_tool5 = load_molecules(uploaded_file_tool5)

        if molecules_tool5:  # Only proceed if molecules_tool5 is not empty
            st.write(f"Number of molecules loaded: {len(molecules_tool5)}")

            threshold_tool5 = st.slider("Set MCS Threshold", 0.0, 1.0, 0.8, 0.05)

            mcs1_tool5, mcs2_tool5, mcs3_tool5, similarity_matrix_tool5, diversity_score_tool5, substructure_freq_tool5 = analyze_molecules(molecules_tool5, threshold_tool5)

            st.subheader("MCS SMARTS Strings Analysis using FMCS Algorithm")
            if mcs1_tool5:
                st.write(f"MCS1 SMARTS: {mcs1_tool5.smartsString}")
            if mcs2_tool5:
                st.write(f"MCS2 SMARTS (Threshold {threshold_tool5}): {mcs2_tool5.smartsString}")
            if mcs3_tool5:
                st.write(f"MCS3 SMARTS (Ring Matches Ring Only): {mcs3_tool5.smartsString}")

            # Display Molecular Diversity Score
            if diversity_score_tool5 is not None:
                st.subheader("Molecular Diversity Score")
                st.write(f"Diversity Score (Mean Similarity): {diversity_score_tool5:.2f}")

            results_text_tool5 = "\n".join([
                f"MCS1 SMARTS: {mcs1_tool5.smartsString if mcs1_tool5 else 'N/A'}",
                f"MCS2 SMARTS (Threshold {threshold_tool5}): {mcs2_tool5.smartsString if mcs2_tool5 else 'N/A'}",
                f"MCS3 SMARTS (Ring Matches Ring Only): {mcs3_tool5.smartsString if mcs3_tool5 else 'N/A'}"
            ])

            st.download_button("Download MCS Data", data=results_text_tool5, file_name="mcs_results.txt")

            # Checkbox for Similarity Matrix plot
            if st.checkbox("Show Similarity Matrix"):
                if similarity_matrix_tool5 is not None:
                    st.subheader("Similarity Matrix")
                    fig_tool5, ax_tool5 = plt.subplots()
                    sns.heatmap(similarity_matrix_tool5, cmap="coolwarm", annot=True, fmt=".2f", ax=ax_tool5)
                    ax_tool5.set_title("Molecular Similarity Matrix")
                    st.pyplot(fig_tool5)

            # Checkbox for Substructure Frequency Analysis plot
            if st.checkbox("Show Substructure Frequency Analysis"):
                if substructure_freq_tool5:
                    st.subheader("Substructure Frequency Analysis")

                    # Rank substructures by frequency and assign numeric labels
                    substructure_df_tool5 = pd.DataFrame(list(substructure_freq_tool5.items()), columns=["Substructure", "Frequency"])
                    substructure_df_tool5 = substructure_df_tool5.sort_values(by="Frequency", ascending=False)
                    substructure_df_tool5["Rank"] = range(1, len(substructure_df_tool5) + 1)

                    # Horizontal bar plot for substructure frequency with numeric labels
                    fig_tool5_subs, ax_tool5_subs = plt.subplots(figsize=(12, 8))
                    sns.barplot(x="Frequency", y="Rank", data=substructure_df_tool5, ax=ax_tool5_subs, palette="Spectral")
                    ax_tool5_subs.set_title("Frequency of Substructures", fontsize=16)
                    ax_tool5_subs.set_xlabel("Frequency", fontsize=14)
                    ax_tool5_subs.set_ylabel("Substructure Rank", fontsize=14)
                    for p in ax_tool5_subs.patches:
                        ax_tool5_subs.annotate(f'{p.get_width():.2f}', (p.get_width(), p.get_y() + p.get_height() / 2), 
                                               ha='left', va='center', fontsize=10)
                    st.pyplot(fig_tool5_subs)

                    # Donut chart for substructure frequency distribution with smaller labels
                    fig_tool5_donut, ax_tool5_donut = plt.subplots(figsize=(8, 6))
                    wedges, texts, autotexts = ax_tool5_donut.pie(
                        substructure_df_tool5["Frequency"], 
                        labels=substructure_df_tool5["Rank"], 
                        autopct='%1.1f%%', startangle=140, 
                        colors=sns.color_palette("Spectral", len(substructure_df_tool5)),
                        wedgeprops=dict(width=0.3)
                    )
                    # Set smaller font size for labels
                    for text in texts:
                        text.set_fontsize(8)
                    for autotext in autotexts:
                        autotext.set_fontsize(8)
                    ax_tool5_donut.set_title("Substructure Frequency Distribution", fontsize=16)
                    st.pyplot(fig_tool5_donut)
                    

#with tab7:
#    st.header("Tool 7: Screengenix-ML")
#elif tool_selection == "Screengenix-ML":
#    st.header("Screengenix-ML")
# Tool 7: Screengenix-ML
elif selected_tool == "Tool 7: ScreenGen-ML":
    st.markdown("<h4 style='color: #3498db;'>Tool 7: Screengenix-ML</h4>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 18px;'>Perform machine learning-based compound screening.</p>", unsafe_allow_html=True)
    
    # Data Processing Information
    st.markdown("---")
    st.markdown("### Data Processing Information")
    st.info("For large datasets, consider using deep learning models for better performance. If your dataset is large, the processing might take a while. Ensure your dataset is preprocessed and includes columns: 'smiles' and 'active'. This tool supports various ML and DL models for ligand-based screening.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="screengenix_ml")
    if uploaded_file is not None:
        chembl_df = pd.read_csv(uploaded_file)
        st.write("Shape of the uploaded dataframe:", chembl_df.shape)
        st.dataframe(chembl_df.head())

        # Standardize column names to lowercase
        chembl_df.columns = [col.lower() for col in chembl_df.columns]
        
        # Define required columns
        required_columns = ["smiles", "pic50"]
        missing_columns = [col for col in required_columns if col not in chembl_df.columns]
        
        # Check for missing columns
        if missing_columns:
            st.warning(f"**Warning:** Missing columns: {', '.join(missing_columns)}. Please upload a file with the required columns.")
            st.stop()
        
        # If 'pic50' is present, use it directly
        if 'pic50' in chembl_df.columns:
            st.write("Using existing pIC50 values.")
        # If 'ic50' is present, convert it to pIC50
        elif 'ic50' in chembl_df.columns:
            chembl_df["pic50"] = -np.log10(chembl_df["ic50"])
            st.write("Converted IC50 to pIC50.")
        else:
            st.warning("**Warning:** Neither 'pic50' nor 'ic50' columns found. Please provide one of these columns.")
            st.stop()

        # Use the derived or existing pic50 for active determination
        chembl_df["active"] = chembl_df["pic50"].apply(lambda x: 1 if x > 5.5 else 0)

        # Check the distribution of the 'active' column
        class_distribution = chembl_df['active'].value_counts()
        if len(class_distribution) < 2:
            st.warning(f"**Warning:** Insufficient class diversity: The dataset contains only one class: {class_distribution.index[0]}. Please provide a dataset with at least two classes (True & False or 0 & 1).")
            st.stop()

        # Updated fingerprint methods
        fingerprint_methods = ["maccs", "morgan2", "morgan3", "rdkit", "topological"]
        method = st.selectbox("Choose Fingerprint Method", fingerprint_methods)
        st.write(f"Using {method} fingerprint")
        chembl_df['fingerprint'] = chembl_df['smiles'].apply(lambda x: smiles_to_fp(x, method=method))
        
        try:
            # Convert 'fingerprint' column to a NumPy array
            X = np.array(chembl_df['fingerprint'].tolist())
            y = chembl_df['active']

            # Ensure all values are numeric
            X = np.nan_to_num(X)

            # Check for correct shapes
            if X.ndim != 2:
                raise ValueError("Feature matrix X is not 2-dimensional.")
            if len(y) != X.shape[0]:
                raise ValueError("Mismatch between the number of samples in X and y.")

        except Exception as e:
            st.warning(f"**Warning:** Error in processing fingerprints: {e}")
            st.stop()
        
        if X.size == 0:
            st.warning("**Warning:** The feature matrix X is empty. Please check the input data.")
            st.stop()
        
        # Train-test split
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Model selection and training
        selected_models = st.multiselect("Select ML Models", [
            "Logistic Regression", 
            "K-Nearest Neighbors", 
            "Decision Tree", 
            "Random Forest", 
            "Gradient Boosting", 
            "AdaBoost", 
            "Extra Trees", 
            "HistGradientBoosting",
            "Support Vector Machine",    
            "XGBoost"
        ])
        
        model_dict = {
            "Logistic Regression": LogisticRegression(),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "AdaBoost": AdaBoostClassifier(),
            "Extra Trees": ExtraTreesClassifier(),
            "HistGradientBoosting": HistGradientBoostingClassifier(),
            "Support Vector Machine": SVC(),
            "XGBoost": XGBClassifier()
        }

        # Check if any model is selected
        if not selected_models:
            st.warning("**Warning:** Please select at least one machine learning model.")
            st.stop()

        results = {}  # Dictionary to store results for each model

        for selected_model in selected_models:
            model = model_dict[selected_model]
            try:
                # Train the model
                accuracy, f1, precision, recall, auc_score, fpr, tpr = model_training_and_validation(model, (train_x, test_x, train_y, test_y))

                results[selected_model] = {
                    "accuracy": accuracy,
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                    "auc_score": auc_score,
                    "fpr": fpr,
                    "tpr": tpr
                }
                
                st.write(f"### {selected_model} Results")
                st.write(f"**Accuracy:** {accuracy:.2f}")
                st.write(f"**F1 Score:** {f1:.2f}")
                st.write(f"**Precision:** {precision:.2f}")
                st.write(f"**Recall:** {recall:.2f}")

            except ValueError as e:
                st.warning(f"**Warning:** Model training error for {selected_model}: {e}")
                st.stop()

        # Plot combined ROC curve if multiple models are selected
        plot_roc_curve07(results)



#with tab8:
#    st.header("Tool 8: Screengenix-DL")
#elif tool_selection == "Screengenix-DL":
#    st.header("Screengenix-DL")
# Tool 8: Screengenix-DL
elif selected_tool == "Tool 8: ScreenGen-DL":
    st.markdown("<h4 style='color: #3498db;'>Tool 8: Screengenix-DL</h4>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 18px;'>Perform deep learning-based compound screening.</p>", unsafe_allow_html=True)
    
    # Data Processing Information
    st.markdown("---")
    st.markdown("### Data Processing Information")
    st.info("For large datasets, consider using deep learning models for better performance. If your dataset is large, the processing might take a while. Ensure your dataset is preprocessed and includes columns: 'smiles' and 'active'. This tool supports various ML and DL models for ligand-based screening.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="screengenix_dl")
    if uploaded_file is not None:
        chembl_df = pd.read_csv(uploaded_file)
        st.write("Shape of the uploaded dataframe:", chembl_df.shape)
        st.dataframe(chembl_df.head())

        # Standardize column names to lowercase
        chembl_df.columns = [col.lower() for col in chembl_df.columns]
        
        # Define required columns in lowercase
        required_columns = ["smiles", "pic50"]
        missing_columns = [col for col in required_columns if col not in chembl_df.columns]
        
        # Check for missing columns
        if missing_columns:
            st.error(f"Missing columns: {', '.join(missing_columns)}. Please upload a file with the required columns.")
            st.stop()
        
        # Update to use lowercase 'pic50'
        chembl_df["active"] = chembl_df["pic50"].apply(lambda x: 1 if x > 5.5 else 0)

        method = st.selectbox("Choose Fingerprint Method", ["maccs", "morgan2", "morgan3"])
        st.write(f"Using {method} fingerprint")
        chembl_df['fingerprint'] = chembl_df['smiles'].apply(lambda x: smiles_to_fp(x, method=method))

        try:
            X = np.array(chembl_df['fingerprint'].tolist())
            y = chembl_df['active']
        except Exception as e:
            st.error(f"Error in processing fingerprints: {e}")
            st.stop()

        if X.size == 0:
            st.error("The feature matrix X is empty. Please check the input data.")
            st.stop()

        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

        selected_dl_model = st.selectbox("Select DL Model", ["Neural Network (NN)", "Convolutional Neural Network (CNN)", "LSTM", "GRU", "RNN", "Transformer", "Attention", "Residual CNN"])

        model_dict_dl = {
            "Neural Network (NN)": build_nn_model(X.shape[1]),
            "Convolutional Neural Network (CNN)": build_cnn_model(X.shape[1]),
            "LSTM": build_lstm_model(X.shape[1]),
            "GRU": build_gru_model(X.shape[1]),
            "RNN": build_rnn_model(X.shape[1]),
            "Transformer": build_transformer_model(X.shape[1]),
            "Attention": build_attention_model(X.shape[1]),
            "Residual CNN": build_residual_cnn_model(X.shape[1])
        }

        model_dl = model_dict_dl[selected_dl_model]
        accuracy, f1, precision, recall, auc_score, fpr, tpr = deep_learning_model_training(model_dl, train_x, train_y, test_x, test_y)

        st.write(f"### {selected_dl_model} Results")
        st.write(f"**Accuracy:** {accuracy:.2f}")
        st.write(f"**F1 Score:** {f1:.2f}")
        st.write(f"**Precision:** {precision:.2f}")
        st.write(f"**Recall:** {recall:.2f}")
        plot_roc_curve(fpr, tpr, auc_score, selected_dl_model)


# Tool 9
elif selected_tool == "Tool 9: TargetPredictor":
    # Tool 9 content
    st.markdown("<h4 style='color: #3498db;'> Tool 9: TargetPredictor </h4>", unsafe_allow_html=True)
    st.markdown("""
    This tool demonstrates interactions between pharmacodynamic components and targets.
    """)
    st.markdown("---")
    
    # Main page for user input
    uploaded_file = st.file_uploader("Upload a CSV file containing your component-target data", type=["csv"])
    
    # Button to load example data
    if st.button("Load Example Data"):
        url = "https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/ExampleData_New.csv"
        df = pd.read_csv(url)
        st.write("Example data loaded successfully:")
        st.write(df.head())

        # Run analysis automatically after loading example data
        if st.button("Run Analysis"):
            G = build_knowledge_graph(df)
            if G:
                fig = plot_interactive_knowledge_graph(G)
                st.plotly_chart(fig)

            X, y, relation_encoded = preprocess_data(df)
            if X is not None and y is not None:
                display_results(df, X, y, relation_encoded)

    def build_knowledge_graph(df):
        try:
            G = nx.DiGraph()
            for _, row in df.iterrows():
                G.add_edge(row['Compound'], row['Target'], relation=row['Relation'])
            return G
        except KeyError as e:
            st.error(f"Missing expected column in the dataset: {e}")
            return None
        except Exception as e:
            st.error(f"An error occurred while building the knowledge graph: {e}")
            return None

    def encode_node_features(node_labels):
        try:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            one_hot_features = encoder.fit_transform(np.array(node_labels).reshape(-1, 1))
            return one_hot_features
        except Exception as e:
            st.error(f"Error during one-hot encoding: {e}")
            return None

    def preprocess_data(df):
        try:
            # Encoding categorical variables
            label_encoder = LabelEncoder()
            df['Compound'] = label_encoder.fit_transform(df['Compound'])
            df['Target'] = label_encoder.fit_transform(df['Target'])
            
            # One-Hot Encoding of the 'Relation' column
            one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            relation_encoded = one_hot_encoder.fit_transform(df[['Relation']])
            
            # Features and labels
            X = df[['Compound', 'Target']].values
            y = df['Relation'].values
            
            # Scaling features
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            
            return X, y, relation_encoded
        except KeyError as e:
            st.error(f"Missing expected column in the dataset for preprocessing: {e}")
            return None, None, None
        except Exception as e:
            st.error(f"An error occurred during preprocessing: {e}")
            return None, None, None

    def display_classification_report(report):
        report_df = pd.DataFrame(report).transpose()
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(report_df.columns),
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[report_df[col] for col in report_df.columns],
                       fill_color='lavender',
                       align='left'))
        ])
        fig.update_layout(title="Classification Report")
        st.plotly_chart(fig)

    def plot_precision_recall_curve(y_test_bin, y_prob):
        fig = go.Figure()
        colors = px.colors.qualitative.Plotly
        
        for i in range(y_prob.shape[1]):
            precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_prob[:, i])
            fig.add_trace(go.Scatter(
                x=recall,
                y=precision,
                mode='lines',
                name=f'Class {i} Precision-Recall curve',
                line=dict(width=2, color=colors[i % len(colors)])
            ))
        
        fig.update_layout(
            title='Precision-Recall Curve',
            xaxis_title='Recall',
            yaxis_title='Precision',
            legend_title='Classes'
        )
        st.plotly_chart(fig)

    def plot_confusion_matrix(cm, labels):
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale='Viridis',
            colorbar=dict(title='Count'),
            zmin=0,
            zmax=np.max(cm)
        ))
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='True',
            xaxis=dict(tickmode='array', tickvals=list(range(len(labels))), ticktext=labels),
            yaxis=dict(tickmode='array', tickvals=list(range(len(labels))), ticktext=labels)
        )
        st.plotly_chart(fig)

    def plot_feature_importances(importance_df):
        # Define colors for each feature
        color_map = {
            'Compound': 'royalblue',
            'Target': 'tomato'
        }
        
        # Map colors to features
        importance_df['Color'] = importance_df['Feature'].map(color_map)
        
        fig = go.Figure(data=go.Bar(
            x=importance_df['Importance'],
            y=importance_df['Feature'],
            orientation='h',
            marker=dict(color=importance_df['Color'])
        ))
        fig.update_layout(
            title='Feature Importances',
            xaxis_title='Importance',
            yaxis_title='Feature'
        )
        st.plotly_chart(fig)

    def plot_interactive_knowledge_graph(G):
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_y.append(y0)
            edge_y.append(y1)
            
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        fig = go.Figure()
        
        # Add edges
        for i in range(0, len(edge_x), 2):
            fig.add_trace(go.Scatter(
                x=[edge_x[i], edge_x[i + 1]],
                y=[edge_y[i], edge_y[i + 1]],
                mode='lines',
                line=dict(width=1.5, color='rgba(150, 150, 150, 0.8)'),
                showlegend=False
            ))
            
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=[node for node in G.nodes()],
            textposition='top center',
            marker=dict(size=15, color='royalblue', line=dict(width=2, color='black')),
            showlegend=False,
            hoverinfo='text'
        ))
        
        fig.update_layout(
            title='Interactive Knowledge Graph',
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            showlegend=False
        )
        
        return fig

    def display_results(df, X, y, relation_encoded):
        try:
            # Keep track of original indices
            df_indices = df.index
            
            X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
                X, y, df_indices, test_size=0.2, random_state=42
            )
            
            # Train a Random Forest Classifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Display Accuracy
            st.subheader("Accuracy")
            st.write(f"**Accuracy**: {accuracy_score(y_test, y_pred):.2f}")
            
            # Display Classification Report
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            display_classification_report(report)
            
            st.header("Visualization")
            
            # Create DataFrame for visualization using indices
            predictions_df = pd.DataFrame({
                "Compound": df.loc[test_indices, 'Compound'],
                "Target": df.loc[test_indices, 'Target'],
                "Prediction": y_pred
            })
            
            # Display interactive table using Plotly
            fig = go.Figure(data=[go.Table(
                header=dict(values=list(predictions_df.columns),
                            fill_color='paleturquoise',
                            align='left'),
                cells=dict(values=[predictions_df[col] for col in predictions_df.columns],
                           fill_color='lavender',
                           align='left'))
            ])
            fig.update_layout(title="Top Predictions")
            st.plotly_chart(fig)
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
            labels = np.unique(y)
            plot_confusion_matrix(cm, labels)
            
            # Precision-Recall Curve
            if len(np.unique(y_test)) > 1:
                y_test_bin = label_binarize(y_test, classes=np.unique(y))
                y_prob = model.predict_proba(X_test)
                plot_precision_recall_curve(y_test_bin, y_prob)
            
            # Feature Importances
            importance_df = pd.DataFrame({
                'Feature': ['Compound', 'Target'],
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            
            plot_feature_importances(importance_df)
        
        except Exception as e:
            st.error(f"An error occurred while displaying results: {e}")

    # Main logic
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(df)

        if st.button("Run Analysis"):
            G = build_knowledge_graph(df)
            if G:
                fig = plot_interactive_knowledge_graph(G)
                st.plotly_chart(fig)

            X, y, relation_encoded = preprocess_data(df)
            if X is not None and y is not None:
                display_results(df, X, y, relation_encoded)


#Tool 10
elif selected_tool == "Tool 10: SmartTox":
    st.markdown("<h4 style='color: #3498db;'>Tool 10: SmartTox</h4>", unsafe_allow_html=True)
    st.markdown("""
SmartTox uses machine learning models to predict drug interactions and potential toxicities.
""")
    st.markdown("---")
    # Tabs for user input
    tab1, tab2 = st.tabs(["Upload CSV", "Enter SMILES"])

    with tab2:
        st.subheader("Enter SMILES")
        user_smiles = st.text_area("Enter SMILES (separated by newlines)")
        if user_smiles:
            smiles_list = user_smiles.splitlines()
            if smiles_list:
                # Assuming `df` is defined elsewhere in your code
                model, X_test, y_test = train_gradient_boosting_model(df)
                adr_predictions = predict_adr(smiles_list, model)
                
                descriptor_df = compute_descriptors(smiles_list)
                graph_signature_df = compute_graph_signatures(smiles_list)
                combined_df = pd.concat([descriptor_df, graph_signature_df], axis=1)
                combined_df = combined_df.apply(pd.to_numeric, errors='coerce').fillna(0)
                
                st.subheader("Computed Molecular Descriptors and Graph Signatures")
                st.write(combined_df)
                
                pk_scores = calculate_pharmacokinetic_scores(smiles_list)  
                pk_df = pd.DataFrame(pk_scores)
                st.subheader("Pharmacokinetic Scores")
                st.write(pk_df)
                
                st.subheader("Pharmacokinetic Scores Distribution")
                fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True)
                sns.histplot(pk_df['AUC'], kde=True, ax=axes[0], color='skyblue', bins=20).set_title("AUC Distribution")
                sns.histplot(pk_df['Clearance'], kde=True, ax=axes[1], color='salmon', bins=20).set_title("Clearance Distribution")
                sns.histplot(pk_df['Volume_of_Distribution'], kde=True, ax=axes[2], color='lightgreen', bins=20).set_title("Volume of Distribution Distribution")
                sns.histplot(pk_df['Hydrogen Bond Donors'], kde=True, ax=axes[3], color='orange', bins=20).set_title("Hydrogen Bond Donors Distribution")
                plt.savefig("pharmacokinetic_scores_distribution.png", dpi=300)
                st.pyplot(fig)
        
                hazards = hazard_identification(smiles_list)
                hazard_df = pd.DataFrame(hazards)
                st.subheader("Hazard Identification")
                st.write(hazard_df)
                
                st.subheader("Hazard Distribution")
                plt.figure(figsize=(8, 6))
                sns.countplot(x='Hazard', data=hazard_df, palette='viridis')
                plt.title('Hazard Categories Distribution')
                plt.savefig("hazard_distribution.png", dpi=300)
                st.pyplot(plt)
        
                toxicophore_scores = calculate_toxicophore_scores(smiles_list)
                toxicophore_df = pd.DataFrame(toxicophore_scores)
                st.subheader("Toxicophore Scores")
                st.write(toxicophore_df)
                
                st.subheader("Toxicophore Scores and Risk Levels")
                fig, axes = plt.subplots(1, 2, figsize=(18, 6))
                
                # Increase the distance between the plots
                plt.subplots_adjust(wspace=0.4)
                
                # Violin plot combined with swarm plot for Toxicophore Scores
                palette = sns.color_palette("cubehelix", n_colors=5)
                palette[1] = '#FF6F61'  # Change the second color to a more suitable one
                
                sns.violinplot(y='Toxicophore Score', data=toxicophore_df, ax=axes[0], inner=None, palette=palette)
                sns.swarmplot(y='Toxicophore Score', data=toxicophore_df, ax=axes[0], color='k', alpha=0.7)
                axes[0].set_title("Toxicophore Scores Distribution")
                
                # Count plot for Toxicophore Risk Levels
                sns.countplot(y='Risk Level', data=toxicophore_df, ax=axes[1], palette='pastel')
                axes[1].set_title("Toxicophore Risk Levels")
                
                plt.savefig("toxicophore_scores_risk_levels.png", dpi=300)
                st.pyplot(fig)
        
                admet_properties = calculate_admet_properties(smiles_list)
                admet_df = pd.DataFrame(admet_properties)
                st.subheader("ADMET Properties")
                st.write(admet_df)
                
                st.subheader("ADMET Properties Distribution")
                fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharex=True)
                sns.histplot(admet_df['Molecular Weight'], kde=True, ax=axes[0, 0], color='orange', bins=20).set_title("Molecular Weight Distribution")
                sns.histplot(admet_df['LogP'], kde=True, ax=axes[0, 1], color='blue', bins=20).set_title("LogP Distribution")
                sns.histplot(admet_df['Hydrogen Bond Donors'], kde=True, ax=axes[0, 2], color='red', bins=20).set_title("Hydrogen Bond Donors Distribution")
                sns.histplot(admet_df['Hydrogen Bond Acceptors'], kde=True, ax=axes[1, 0], color='green', bins=20).set_title("Hydrogen Bond Acceptors Distribution")
                sns.histplot(admet_df['Rotatable Bonds'], kde=True, ax=axes[1, 1], color='purple', bins=20).set_title("Rotatable Bonds Distribution")
                sns.histplot(admet_df['LogP'], kde=True, ax=axes[1, 2], color='blue', bins=20).set_title("LogP Distribution")
                plt.savefig("admet_properties_distribution.png", dpi=300)
                st.pyplot(fig)
        
                st.subheader("ADMET Properties Heatmap")
                numeric_admet_df = admet_df.select_dtypes(include=[np.number])
                if not numeric_admet_df.empty:
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(numeric_admet_df.corr(), annot=True, cmap='coolwarm', center=0)
                    plt.title("Heatmap of ADMET Properties")
                    plt.savefig("admet_properties_heatmap.png", dpi=300)
                    st.pyplot(plt)
                else:
                    st.write("No numeric ADMET properties available for heatmap.")

                admet_properties = calculate_admet_properties2(smiles_list)
                admet_df = pd.DataFrame(admet_properties)
                st.subheader("ADMET Properties with Additional Parameters")
                st.write(admet_df)
            
                # Numeric properties distribution
                fig, axes = plt.subplots(3, 3, figsize=(18, 18))
                palette = sns.color_palette("Set2", n_colors=9)
                
                sns.histplot(admet_df['Molecular Weight'], kde=True, ax=axes[0, 0], color=palette[0]).set_title("Molecular Weight Distribution")
                sns.histplot(admet_df['LogP'], kde=True, ax=axes[0, 1], color=palette[1]).set_title("LogP Distribution")
                sns.histplot(admet_df['Hydrogen Bond Donors'], kde=True, ax=axes[0, 2], color=palette[2]).set_title("Hydrogen Bond Donors Distribution")
                sns.histplot(admet_df['Hydrogen Bond Acceptors'], kde=True, ax=axes[1, 0], color=palette[3]).set_title("Hydrogen Bond Acceptors Distribution")
                sns.histplot(admet_df['Rotatable Bonds'], kde=True, ax=axes[1, 1], color=palette[4]).set_title("Rotatable Bonds Distribution")
                sns.histplot(admet_df['LogP'], kde=True, ax=axes[1, 2], color=palette[5]).set_title("LogP Distribution (Replicated)")
                sns.histplot(admet_df['Half-life'], kde=False, ax=axes[2, 0], color=palette[6]).set_title("Half-life Distribution")
                sns.histplot(admet_df['Phytoplankton Toxicity'], kde=False, ax=axes[2, 1], color=palette[7]).set_title("Phytoplankton Toxicity")
                sns.histplot(admet_df['Biodegradation'], kde=False, ax=axes[2, 2], color=palette[8]).set_title("Biodegradation")
                
                for ax in axes.flat:
                    ax.set_xlabel(ax.get_xlabel(), fontsize=12)
                    ax.set_ylabel(ax.get_ylabel(), fontsize=12)
                    ax.tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Categorical properties distribution
                fig, axes = plt.subplots(3, 3, figsize=(18, 18))
                
                sns.countplot(data=admet_df, x='HIA', ax=axes[0, 0], palette='Set2').set_title("Human Intestinal Absorption")
                sns.countplot(data=admet_df, x='HOB', ax=axes[0, 1], palette='Set2').set_title("Human Oral Bioavailability")
                sns.countplot(data=admet_df, x='Caco-2', ax=axes[0, 2], palette='Set2').set_title("Caco-2 Permeability")
                sns.countplot(data=admet_df, x='BBB Penetration', ax=axes[1, 0], palette='Set2').set_title("Blood-Brain Barrier Penetration")
                sns.countplot(data=admet_df, x='BPB Penetration', ax=axes[1, 1], palette='Set2').set_title("Blood-Placenta Barrier Penetration")
                sns.countplot(data=admet_df, x='P-gp Substrate', ax=axes[1, 2], palette='Set2').set_title("P-gp Substrate")
                sns.countplot(data=admet_df, x='PPB', ax=axes[2, 0], palette='Set2').set_title("Plasma Protein Binding")
                sns.countplot(data=admet_df, x='Ames Test', ax=axes[2, 1], palette='Set2').set_title("Ames Test")
                sns.countplot(data=admet_df, x='Carcinogenicity', ax=axes[2, 2], palette='Set2').set_title("Carcinogenicity")
                
                for ax in axes.flat:
                    ax.tick_params(axis='x', rotation=45)
                    ax.set_xlabel(ax.get_xlabel(), fontsize=12)
                    ax.set_ylabel(ax.get_ylabel(), fontsize=12)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                #adr_predictions = predict_adr(smiles_list, model)
                # Create a DataFrame for the predictions with interpretations
                prediction_results = pd.DataFrame({'SMILES': smiles_list, 'ADR Prediction': adr_predictions})
                # Apply interpretation to each prediction
                prediction_results['Interpretation'] = prediction_results['ADR Prediction'].apply(interpret_score)
                # Display results
                st.subheader("ADR Predictions using Gradient Boosting")
                st.write(prediction_results)

                #st.subheader("ADR Classification Report")
                #y_pred = model.predict(X_test)
                #report = classification_report(y_test, y_pred, output_dict=True)
                #report_df = pd.DataFrame(report).transpose()
                #report_df = report_df.style.background_gradient(cmap='RdYlGn', subset=pd.IndexSlice[:, ['precision', 'recall', 'f1-score']])
                #st.write(report_df)

                # Additional model evaluation visualizations
                #def plot_roc_curve(model, X_test, y_test):
                    #plt.figure(figsize=(8, 6))
                    #y_pred_prob = model.predict_proba(X_test)[:, 1]
                    #fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
                    #roc_auc = auc(fpr, tpr)
                    #plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
                    #plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    #plt.xlim([0.0, 1.0])
                    #plt.ylim([0.0, 1.05])
                    #plt.xlabel('False Positive Rate')
                    #plt.ylabel('True Positive Rate')
                    #plt.title('Receiver Operating Characteristic (ROC)')
                    #plt.legend(loc="lower right")
                    #plt.savefig("roc_curve.png", dpi=300)
                    #st.pyplot(plt)
                
                #def plot_precision_recall_curve(model, X_test, y_test):
                    #plt.figure(figsize=(8, 6))
                    #y_pred_prob = model.predict_proba(X_test)[:, 1]
                    #precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
                    #plt.plot(recall, precision, color='blue', lw=2, label=f"PR curve (AUC = {roc_auc_score(y_test, y_pred_prob):.2f})")
                    #plt.xlabel('Recall')
                    #plt.ylabel('Precision')
                    #plt.title('Precision-Recall Curve')
                    #plt.legend(loc="lower left")
                    #plt.savefig("precision_recall_curve.png", dpi=300)
                    #st.pyplot(plt)
                
                #def plot_confusion_matrix(y_test, y_pred):
                    #conf_matrix = confusion_matrix(y_test, y_pred)
                    #plt.figure(figsize=(8, 6))
                    #sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
                    #plt.title('Confusion Matrix')
                    #plt.ylabel('Actual Label')
                    #plt.xlabel('Predicted Label')
                    #plt.savefig("confusion_matrix.png", dpi=300)
                    #st.pyplot(plt)

                #plot_roc_curve(model, X_test, y_test)
                #plot_precision_recall_curve(model, X_test, y_test)
                #plot_confusion_matrix(y_test, y_pred)
        else:
            st.warning("Please enter SMILES strings or upload a CSV file.")

    # In the "Upload CSV" tab
    with tab1:
        #st.header("Upload CSV")
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file:
            df_uploaded = pd.read_csv(uploaded_file)
            smiles_column_name = next((col for col in df_uploaded.columns if col.lower() in ['smiles', 'smile', "SMILES", "SMILE", "Smiles", "Smile"]), None)
            if smiles_column_name:
                # Remove any rows with missing SMILES values
                df_uploaded = df_uploaded.dropna(subset=[smiles_column_name])
                smiles_list = df_uploaded[smiles_column_name].tolist()
            else:
                st.error("CSV file must contain a column named 'SMILES', 'smiles', or similar.")
        else:
            smiles_list = []


    if smiles_list:
        # Assuming `df` is defined elsewhere in your code
        model, X_test, y_test = train_gradient_boosting_model(df)
        adr_predictions = predict_adr(smiles_list, model)
        descriptor_df = compute_descriptors(smiles_list)
        graph_signature_df = compute_graph_signatures(smiles_list)
        combined_df = pd.concat([descriptor_df, graph_signature_df], axis=1)
        combined_df = combined_df.apply(pd.to_numeric, errors='coerce').fillna(0)

        st.subheader("Computed Molecular Descriptors and Graph Signatures")
        st.write(combined_df)

        pk_scores = calculate_pharmacokinetic_scores(smiles_list)  
        pk_df = pd.DataFrame(pk_scores)
        st.subheader("Pharmacokinetic Scores")
        st.write(pk_df)
        
        st.subheader("Pharmacokinetic Scores Distribution")
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True)
        sns.histplot(pk_df['AUC'], kde=True, ax=axes[0], color='skyblue', bins=20).set_title("AUC Distribution")
        sns.histplot(pk_df['Clearance'], kde=True, ax=axes[1], color='salmon', bins=20).set_title("Clearance Distribution")
        sns.histplot(pk_df['Volume_of_Distribution'], kde=True, ax=axes[2], color='lightgreen', bins=20).set_title("Volume of Distribution Distribution")
        sns.histplot(pk_df['Hydrogen Bond Donors'], kde=True, ax=axes[3], color='orange', bins=20).set_title("Hydrogen Bond Donors Distribution")
        plt.savefig("pharmacokinetic_scores_distribution.png", dpi=300)
        st.pyplot(fig)

        # Run Hazard Identification
        hazards = hazard_identification(smiles_list)
        
        # Convert results to DataFrame
        hazard_df = pd.DataFrame(hazards)
        
        # Streamlit UI
        st.subheader("Hazard Identification")
        st.write(hazard_df)
        
        # Hazard Distribution Plot
        st.subheader("Hazard Distribution")
        plt.figure(figsize=(8, 6))
        sns.countplot(x='Hazard', data=hazard_df, palette='viridis')
        plt.title('Hazard Categories Distribution')
        plt.xlabel('Hazard Level')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig("hazard_distribution.png", dpi=300)
        
        # Display Plot in Streamlit
        st.pyplot(plt)

        toxicophore_scores = calculate_toxicophore_scores(smiles_list)
        toxicophore_df = pd.DataFrame(toxicophore_scores)
        st.subheader("Toxicophore Scores")
        st.write(toxicophore_df)
        
        st.subheader("Toxicophore Scores and Risk Levels")
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        
        # Increase the distance between the plots
        plt.subplots_adjust(wspace=0.4)
        
        # Violin plot combined with swarm plot for Toxicophore Scores
        palette = sns.color_palette("cubehelix", n_colors=5)
        palette[1] = '#FF6F61'  # Change the second color to a more suitable one
        
        sns.violinplot(y='Toxicophore Score', data=toxicophore_df, ax=axes[0], inner=None, palette=palette)
        sns.swarmplot(y='Toxicophore Score', data=toxicophore_df, ax=axes[0], color='k', alpha=0.7)
        axes[0].set_title("Toxicophore Scores Distribution")
        
        # Count plot for Toxicophore Risk Levels
        sns.countplot(y='Risk Level', data=toxicophore_df, ax=axes[1], palette='pastel')
        axes[1].set_title("Toxicophore Risk Levels")
        
        plt.savefig("toxicophore_scores_risk_levels.png", dpi=300)
        st.pyplot(fig)

        admet_properties = calculate_admet_properties(smiles_list)
        admet_df = pd.DataFrame(admet_properties)
        st.subheader("ADMET Properties")
        st.write(admet_df)
        
        st.subheader("ADMET Properties Distribution")
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharex=True)
        sns.histplot(admet_df['Molecular Weight'], kde=True, ax=axes[0, 0], color='orange', bins=20).set_title("Molecular Weight Distribution")
        sns.histplot(admet_df['LogP'], kde=True, ax=axes[0, 1], color='blue', bins=20).set_title("LogP Distribution")
        sns.histplot(admet_df['Hydrogen Bond Donors'], kde=True, ax=axes[0, 2], color='red', bins=20).set_title("Hydrogen Bond Donors Distribution")
        sns.histplot(admet_df['Hydrogen Bond Acceptors'], kde=True, ax=axes[1, 0], color='green', bins=20).set_title("Hydrogen Bond Acceptors Distribution")
        sns.histplot(admet_df['Rotatable Bonds'], kde=True, ax=axes[1, 1], color='purple', bins=20).set_title("Rotatable Bonds Distribution")
        sns.histplot(admet_df['LogP'], kde=True, ax=axes[1, 2], color='blue', bins=20).set_title("LogP Distribution")
        plt.savefig("admet_properties_distribution.png", dpi=300)
        st.pyplot(fig)

        st.subheader("ADMET Properties Heatmap")
        numeric_admet_df = admet_df.select_dtypes(include=[np.number])
        if not numeric_admet_df.empty:
            plt.figure(figsize=(10, 8))
            sns.heatmap(numeric_admet_df.corr(), annot=True, cmap='coolwarm', center=0)
            plt.title("Heatmap of ADMET Properties")
            plt.savefig("admet_properties_heatmap.png", dpi=300)
            st.pyplot(plt)
        else:
            st.write("No numeric ADMET properties available for heatmap.")

        admet_properties = calculate_admet_properties2(smiles_list)
        admet_df = pd.DataFrame(admet_properties)
        st.subheader("ADMET Properties with Additional Parameters")
        st.write(admet_df)
    
        # Numeric properties distribution
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        palette = sns.color_palette("Set2", n_colors=9)
        
        sns.histplot(admet_df['Molecular Weight'], kde=True, ax=axes[0, 0], color=palette[0]).set_title("Molecular Weight Distribution")
        sns.histplot(admet_df['LogP'], kde=True, ax=axes[0, 1], color=palette[1]).set_title("LogP Distribution")
        sns.histplot(admet_df['Hydrogen Bond Donors'], kde=True, ax=axes[0, 2], color=palette[2]).set_title("Hydrogen Bond Donors Distribution")
        sns.histplot(admet_df['Hydrogen Bond Acceptors'], kde=True, ax=axes[1, 0], color=palette[3]).set_title("Hydrogen Bond Acceptors Distribution")
        sns.histplot(admet_df['Rotatable Bonds'], kde=True, ax=axes[1, 1], color=palette[4]).set_title("Rotatable Bonds Distribution")
        sns.histplot(admet_df['LogP'], kde=True, ax=axes[1, 2], color=palette[5]).set_title("LogP Distribution (Replicated)")
        sns.histplot(admet_df['Half-life'], kde=False, ax=axes[2, 0], color=palette[6]).set_title("Half-life Distribution")
        sns.histplot(admet_df['Phytoplankton Toxicity'], kde=False, ax=axes[2, 1], color=palette[7]).set_title("Phytoplankton Toxicity")
        sns.histplot(admet_df['Biodegradation'], kde=False, ax=axes[2, 2], color=palette[8]).set_title("Biodegradation")
        
        for ax in axes.flat:
            ax.set_xlabel(ax.get_xlabel(), fontsize=12)
            ax.set_ylabel(ax.get_ylabel(), fontsize=12)
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Categorical properties distribution
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        
        sns.countplot(data=admet_df, x='HIA', ax=axes[0, 0], palette='Set2').set_title("Human Intestinal Absorption")
        sns.countplot(data=admet_df, x='HOB', ax=axes[0, 1], palette='Set2').set_title("Human Oral Bioavailability")
        sns.countplot(data=admet_df, x='Caco-2', ax=axes[0, 2], palette='Set2').set_title("Caco-2 Permeability")
        sns.countplot(data=admet_df, x='BBB Penetration', ax=axes[1, 0], palette='Set2').set_title("Blood-Brain Barrier Penetration")
        sns.countplot(data=admet_df, x='BPB Penetration', ax=axes[1, 1], palette='Set2').set_title("Blood-Placenta Barrier Penetration")
        sns.countplot(data=admet_df, x='P-gp Substrate', ax=axes[1, 2], palette='Set2').set_title("P-gp Substrate")
        sns.countplot(data=admet_df, x='PPB', ax=axes[2, 0], palette='Set2').set_title("Plasma Protein Binding")
        sns.countplot(data=admet_df, x='Ames Test', ax=axes[2, 1], palette='Set2').set_title("Ames Test")
        sns.countplot(data=admet_df, x='Carcinogenicity', ax=axes[2, 2], palette='Set2').set_title("Carcinogenicity")
        
        for ax in axes.flat:
            ax.tick_params(axis='x', rotation=45)
            ax.set_xlabel(ax.get_xlabel(), fontsize=12)
            ax.set_ylabel(ax.get_ylabel(), fontsize=12)
        
        plt.tight_layout()
        st.pyplot(fig)
                
        # Display results
        #adr_predictions = predict_adr(smiles_list, model)
        # Create a DataFrame for the predictions with interpretations
        prediction_results = pd.DataFrame({'SMILES': smiles_list, 'ADR Prediction': adr_predictions})
        # Apply interpretation to each prediction
        prediction_results['Interpretation'] = prediction_results['ADR Prediction'].apply(interpret_score)
        # Display results
        st.subheader("ADR Predictions using Gradient Boosting")
        st.write(prediction_results)

        #st.subheader("ADR Classification Report")
        #y_pred = model.predict(X_test)
        #report = classification_report(y_test, y_pred, output_dict=True)
        #report_df = pd.DataFrame(report).transpose()
        #report_df = report_df.style.background_gradient(cmap='RdYlGn', subset=pd.IndexSlice[:, ['precision', 'recall', 'f1-score']])
        #st.write(report_df)

        # Additional model evaluation visualizations
        #def plot_roc_curve(model, X_test, y_test):
            #plt.figure(figsize=(8, 6))
            #y_pred_prob = model.predict_proba(X_test)[:, 1]
            #fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
            #roc_auc = auc(fpr, tpr)
            #plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
            #plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            #plt.xlim([0.0, 1.0])
            #plt.ylim([0.0, 1.05])
            #plt.xlabel('False Positive Rate')
            #plt.ylabel('True Positive Rate')
            #plt.title('Receiver Operating Characteristic (ROC)')
            #plt.legend(loc="lower right")
            #st.pyplot(plt)
                
        #def plot_precision_recall_curve(model, X_test, y_test):
            #plt.figure(figsize=(8, 6))
            #y_pred_prob = model.predict_proba(X_test)[:, 1]
            #precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
            #plt.plot(recall, precision, color='blue', lw=2, label=f"PR curve (AUC = {roc_auc_score(y_test, y_pred_prob):.2f})")
            #plt.xlabel('Recall')
            #plt.ylabel('Precision')
            #plt.title('Precision-Recall Curve')
            #plt.legend(loc="lower left")
            #plt.savefig("precision_recall_curve.png", dpi=300)
            #st.pyplot(plt)
                
        #def plot_confusion_matrix(y_test, y_pred):
            #conf_matrix = confusion_matrix(y_test, y_pred)
            #plt.figure(figsize=(8, 6))
            #sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            #plt.title('Confusion Matrix')
            #plt.ylabel('Actual Label')
            #plt.xlabel('Predicted Label')
            #plt.savefig("confusion_matrix.png", dpi=300)
            #st.pyplot(plt)

        #plot_roc_curve(model, X_test, y_test)
        #plot_precision_recall_curve(model, X_test, y_test)
        #plot_confusion_matrix(y_test, y_pred)
    else:
        st.warning("Please enter SMILES strings or upload a CSV file.")

# Tool 11: Drug-Drug Interaction Predictor
elif selected_tool == "Tool 11: DDI-Predictor":
    st.markdown("<h4 style='color: #3498db;'>Tool 11: Drug-Drug Interaction</h4>", unsafe_allow_html=True)
    st.markdown("This tool demonstrates interactions between drugs based on their SMILES representations.")

    # Tool: 11 Functions
    # Function to download and save the model locally
    def download_model(url, local_path):
        response = requests.get(url)
        response.raise_for_status()
        with open(local_path, 'wb') as f:
            f.write(response.content)
    
    # Local path to save the model
    model_path = 'xgb_model.pkl'
    xgb_model_url = 'https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/DrugBank_Dataset/xgb_model.pkl'
    
    # Check if model is already downloaded
    if not os.path.exists(model_path):
        download_model(xgb_model_url, model_path)
    
    # Load the XGBoost model from the local file
    xgb_model = joblib.load(model_path)
    
    # Function to download CSV from GitHub
    def download_csv(url):
        response = requests.get(url)
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text))
        
    # File upload options
    st.sidebar.header("Upload CSV Files")
    uploaded_file_drug1 = st.file_uploader("Upload CSV file for Drug1 List", type=["csv"])
    uploaded_file_drug2 = st.file_uploader("Upload CSV file for Drug2 List", type=["csv"])
    
    # Function to extract SMILES column and labels from CSV file
    def extract_smiles_and_labels(df):
        # List of possible SMILES column names (case-insensitive)
        possible_smiles_cols = ['SMILES', 'Smiles', 'smiles', 'SMILE', 'Smile', 'smile']
        label_col = 'Y'
        
        # Find the SMILES column in the dataframe, ignoring case
        smiles_col = next((col for col in df.columns if col.lower() in [s.lower() for s in possible_smiles_cols]), None)
        
        if smiles_col:
            smiles = df[smiles_col].dropna().tolist()
            if label_col in df.columns:
                labels = df[label_col].apply(lambda x: 1 if x > 30 else 0).tolist()
                return smiles, labels
            else:
                # If 'Y' column is missing, return SMILES without labels
                return smiles, None
        st.error("No valid SMILES column found in the uploaded CSV file.")
        return [], None
    
    # Function to display results for SMILES-only data
    def display_results_for_smiles_only(df, X):
        try:
            # Predict with XGBoost model
            y_prob = xgb_model.predict_proba(X)[:, 1]  # Probability of the positive class
            df['Prediction_Score'] = y_prob
            
            st.subheader("Predicted Interaction Scores")
            st.dataframe(df[['smiles_drug1', 'smiles_drug2', 'Prediction_Score']])
            
            # Additional plots can be included here if relevant for SMILES-only data
        except Exception as e:
            st.error(f"An error occurred while displaying results for SMILES-only data: {e}")
    
    # Convert SMILES to DataFrame for prediction
    def smiles_to_dataframe(drug1_smiles, drug2_smiles):
        if isinstance(drug1_smiles, str):
            drug1_smiles = [drug1_smiles]
        if isinstance(drug2_smiles, str):
            drug2_smiles = [drug2_smiles]
        
        # Extract features
        features = extract_molecular_features(drug1_smiles, drug2_smiles)
        
        # Create DataFrame
        df = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(features.shape[1])])
        return df
    
    
    def calculate_tanimoto_similarity(drug1_smiles, drug2_smiles):
        similarities = []
        for smi1 in drug1_smiles:
            mol1 = Chem.MolFromSmiles(smi1)
            fp1 = FingerprintMols.FingerprintMol(mol1)
            for smi2 in drug2_smiles:
                mol2 = Chem.MolFromSmiles(smi2)
                fp2 = FingerprintMols.FingerprintMol(mol2)
                similarity = DataStructs.FingerprintSimilarity(fp1, fp2)
                similarities.append(similarity)
        return similarities
    
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski
    
    def extract_molecular_features(smiles_drug1, smiles_drug2):
        features_drug1 = []
        features_drug2 = []
        
        if isinstance(smiles_drug1, str):
            smiles_drug1 = [smiles_drug1]
        if isinstance(smiles_drug2, str):
            smiles_drug2 = [smiles_drug2]
        
        for smi1, smi2 in zip(smiles_drug1, smiles_drug2):
            mol1 = Chem.MolFromSmiles(smi1)
            mol2 = Chem.MolFromSmiles(smi2)
            
            if mol1 and mol2:
                features_drug1.append([
                    Descriptors.MolWt(mol1),
                    Descriptors.NumRotatableBonds(mol1),
                    Descriptors.NumHDonors(mol1),
                    Descriptors.NumHAcceptors(mol1),
                    rdMolDescriptors.CalcTPSA(mol1),
                    rdMolDescriptors.CalcNumRings(mol1),
                    Lipinski.NumRotatableBonds(mol1),
                    Lipinski.NumHDonors(mol1),
                    Lipinski.NumHAcceptors(mol1),
                    Descriptors.MolLogP(mol1)
                ])
                features_drug2.append([
                    Descriptors.MolWt(mol2),
                    Descriptors.NumRotatableBonds(mol2),
                    Descriptors.NumHDonors(mol2),
                    Descriptors.NumHAcceptors(mol2),
                    rdMolDescriptors.CalcTPSA(mol2),
                    rdMolDescriptors.CalcNumRings(mol2),
                    Lipinski.NumRotatableBonds(mol2),
                    Lipinski.NumHDonors(mol2),
                    Lipinski.NumHAcceptors(mol2),
                    Descriptors.MolLogP(mol2)
                ])
            else:
                features_drug1.append([np.nan] * 10)
                features_drug2.append([np.nan] * 10)
        
        features_drug1 = np.array(features_drug1)
        features_drug2 = np.array(features_drug2)
        
        combined_features = np.hstack([features_drug1, features_drug2])
        #st.write(f"Features combined shape: {combined_features.shape}")
        return combined_features
    
    def encode_node_features(node_labels):
        try:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            one_hot_features = encoder.fit_transform(np.array(node_labels).reshape(-1, 1))
            return one_hot_features
        except Exception as e:
            st.error(f"Error during one-hot encoding: {e}")
            return None
    
    def preprocess_data(df_pred):
        try:
            # Extract features from the DataFrame
            X = extract_molecular_features(df_pred['smiles_drug1'], df_pred['smiles_drug2'])
            
            if X is None or X.size == 0:
                raise ValueError("Feature extraction returned None or empty array.")
            
            #st.write(f"Extracted features shape: {X.shape}")
            
            # Handle missing values if necessary
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(X)
            
            # Scaling features
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            
            #st.write(f"Preprocessed features shape: {X.shape}")
            
            return X
    
        except Exception as e:
            st.error(f"An error occurred during preprocessing: {e}")
            return None
    
    def display_classification_report(report):
        report_df = pd.DataFrame(report).transpose()
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(report_df.columns),
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[report_df[col] for col in report_df.columns],
                       fill_color='lavender',
                       align='left'))
        ])
        fig.update_layout(title="Classification Report")
        st.plotly_chart(fig)
    
    def plot_precision_recall_curve(y_true, y_prob):
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='Precision-Recall curve'))
        fig.update_layout(title='Precision-Recall Curve', xaxis_title='Recall', yaxis_title='Precision')
        st.plotly_chart(fig)
    
    def plot_confusion_matrix(cm, labels):
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale='Viridis',
            colorbar=dict(title='Count'),
            zmin=0,
            zmax=np.max(cm)
        ))
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='True',
            xaxis=dict(tickmode='array', tickvals=list(range(len(labels))), ticktext=labels),
            yaxis=dict(tickmode='array', tickvals=list(range(len(labels))), ticktext=labels)
        )
        st.plotly_chart(fig)
    
    def plot_feature_importances(importance_df):
        # Define colors for each feature
        color_map = {
            'Drug1': 'royalblue',
            'Drug2': 'tomato'
        }
        
        # Map colors to features
        importance_df['Color'] = importance_df['Feature'].map(color_map)
        
        fig = go.Figure(data=go.Bar(
            x=importance_df['Importance'],
            y=importance_df['Feature'],
            orientation='h',
            marker=dict(color=importance_df['Color'])
        ))
        fig.update_layout(
            title='Feature Importances',
            xaxis_title='Importance',
            yaxis_title='Feature'
        )
        st.plotly_chart(fig)
    
    def plot_interactive_knowledge_graph(G):
        pos = nx.spring_layout(G, k=0.3, iterations=100)  # Adjust layout parameters as needed
        
        # Extract positions
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_y.append(y0)
            edge_y.append(y1)
        
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        # Define color palette and transparency based on score
        node_color = []
        node_opacity = []
        for i, node in enumerate(G.nodes()):
            score = i / len(G.nodes())  # Assuming score is related to node index; adjust as needed
            node_color.append(f'rgba(31, 119, 180, {0.3 + 0.7 * score})')  # Color with transparency
            node_opacity.append(0.3 + 0.7 * score)
        
        # Define edge animation frames
        frames = []
        num_frames = 20
        for i in range(num_frames):
            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                mode='lines',
                line=dict(width=2, color='rgba(150, 150, 150, 0.8)', dash='dash'),
                opacity=0.8
            )
            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                text=[node for node in G.nodes()],
                textposition='top center',
                textfont=dict(color='rgba(0, 0, 0, 0.1)'),  # Set text color to semi-transparent
                marker=dict(size=15 + 5 * (i % 5), color=node_color, line=dict(width=2, color='black')),
                hoverinfo='text'
            )
            frames.append(go.Frame(
                data=[edge_trace, node_trace],
                name=f'Frame {i}'
            ))
    
        # Create the initial plot
        fig = go.Figure()
    
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(width=2, color='rgba(150, 150, 150, 0.8)'),
            showlegend=False
        ))
    
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=[node for node in G.nodes()],
            textposition='top center',
            textfont=dict(color='rgba(0, 0, 0, 0.2)'),  # Set text color to semi-transparent
            marker=dict(size=15, color=node_color, opacity=node_opacity, line=dict(width=2, color='black')),
            showlegend=False,
            hoverinfo='text'
        ))
    
        # Add frames to the figure
        fig.frames = frames
    
        # Add animation settings
        fig.update_layout(
            title='Interactive Knowledge Graph with Animation',
            title_x=0.5,
            showlegend=False,
            updatemenus=[
                dict(
                    type='buttons',
                    showactive=False,
                    buttons=[
                        dict(
                            label='Play',
                            method='animate',
                            args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True, mode='immediate')]
                        ),
                        dict(
                            label='Pause',
                            method='animate',
                            args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')]
                        )
                    ]
                )
            ]
        )
    
        # Add hover effect for nodes
        fig.update_traces(marker=dict(size=10, opacity=0.8), selector=dict(mode='markers+text'))
        fig.update_layout(
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False)
        )
        
        return fig
    
    def build_knowledge_graph(df):
        try:
            G = nx.DiGraph()
            interactions = []
            for _, row in df.iterrows():
                if 'smiles_drug1' in row and 'smiles_drug2' in row:
                    G.add_edge(row['smiles_drug1'], row['smiles_drug2'])
                    interactions.append({'Drug1': row['smiles_drug1'], 'Drug2': row['smiles_drug2'], 'Interaction': np.random.randint(0, 2)})  # Replace with actual method
            interaction_df = pd.DataFrame(interactions)
            return G, interaction_df
        except KeyError as e:
            st.error(f"Missing expected column in the dataset: {e}")
            return None, None
        except Exception as e:
            st.error(f"An error occurred while building the knowledge graph: {e}")
            return None, None
    
    def plot_interaction_heatmap(interaction_df):
        try:
            # Create a pivot table to represent the interaction matrix
            interaction_matrix = interaction_df.pivot(index="Drug1", columns="Drug2", values="Interaction")
            
            # Plot heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(interaction_matrix, annot=True, cmap="YlGnBu", cbar=True)
            plt.title("Heatmap of Drug-Drug Interactions")
            plt.xlabel("Drug 2")
            plt.ylabel("Drug 1")
            
            # Display in Streamlit
            st.pyplot(plt)
            plt.clf()
            
        except Exception as e:
            st.error(f"An error occurred while plotting the heatmap: {e}")
    
    def plot_interaction_counts(interaction_df):
        try:
            # Count the number of interactions for each drug
            interaction_counts = interaction_df['Drug1'].value_counts().reset_index()
            interaction_counts.columns = ['Drug', 'Count']
            
            # Create bar plot
            fig = px.bar(interaction_counts, x='Drug', y='Count', title="Interaction Counts for Each Drug")
            
            st.plotly_chart(fig)
            
        except Exception as e:
            st.error(f"An error occurred while plotting the interaction counts: {e}")
    
    # Update display_results to include only CNN model
    def display_results(df, X, true_labels):
        try:
            # Predict with XGBoost model
            y_pred = xgb_model.predict(X)
            y_prob = xgb_model.predict_proba(X)[:, 1]  # Probability of the positive class
            df['Prediction'] = y_pred
            
            # Calculate accuracy
            accuracy = accuracy_score(true_labels, y_pred)
            
            st.subheader("Predicted Interactions")
            st.dataframe(df[['smiles_drug1', 'smiles_drug2', 'Prediction']])
            
            st.subheader("Prediction Accuracy")
            st.write(f"Accuracy: {accuracy:.2f}")
    
            # Classification report
            report = classification_report(true_labels, y_pred, output_dict=True)
            #st.subheader("Classification Report")
            display_classification_report(report)
            
            # Calculate confusion matrix
            cm = confusion_matrix(true_labels, y_pred)
            labels = ['No Interaction', 'Interaction']
            #st.write("Confusion Matrix:")
            plot_confusion_matrix(cm, labels)
            
            # Plot precision-recall curve
            #st.write("Precision-Recall Curve:")
            plot_precision_recall_curve(true_labels, y_prob)
            
            # Calculate feature importances
            importances = xgb_model.feature_importances_
            feature_names = [f'feature_{i}' for i in range(len(importances))]
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            })
            importance_df = importance_df.sort_values(by='Importance', ascending=False)
            #st.write("Feature Importances:")
            plot_feature_importances(importance_df)
    
            # Build and plot knowledge graph
            st.subheader("Knowledge Graph")
            G, interaction_df = build_knowledge_graph(df)
            if G:
                fig = plot_interactive_knowledge_graph(G)
                st.plotly_chart(fig)
            
            # Ensure interaction_df is not None before plotting
            if interaction_df is not None:
                #st.subheader("Drug-Drug Interaction Heatmap")
                plot_interaction_heatmap(interaction_df)
                
                #st.subheader("Drug Interaction Counts")
                plot_interaction_counts(interaction_df)
            
        except Exception as e:
            st.error(f"An error occurred while displaying results: {e}")
    
    # Main code block
    if uploaded_file_drug1 and uploaded_file_drug2:
        df_drug1 = pd.read_csv(uploaded_file_drug1)
        df_drug2 = pd.read_csv(uploaded_file_drug2)
    
        drug1_smiles, drug1_labels = extract_smiles_and_labels(df_drug1)
        drug2_smiles, drug2_labels = extract_smiles_and_labels(df_drug2)
    
        if drug1_smiles and drug2_smiles:
            # Assuming one-to-one correspondence between drugs
            df_pred = pd.DataFrame({
                'smiles_drug1': drug1_smiles,
                'smiles_drug2': drug2_smiles
            })
    
            # Generate features
            X = preprocess_data(df_pred)
    
            if X is not None:
                # Check if labels are available for accuracy evaluation
                if drug1_labels is not None:
                    display_results(df_pred, X, drug1_labels)
                else:
                    # Display results without labels
                    display_results_for_smiles_only(df_pred, X)


# Tool 12: CellDrugFinder
elif selected_tool == "Tool 12: CellDrugFinder":
    st.markdown("<h4 style='color: #3498db;'>Tool 12: CellDrugFinder</h4>", unsafe_allow_html=True)
    st.markdown("""
        Tissue and Cell Line-specific Essential Drug Prediction.
    """)
    st.markdown("---")
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Predictive Modeling",
        "Drug Analysis",
        "Cell Line Analysis",
        "Tissue Analysis",
        "Statistical Analysis",
        "In-Vitro Prediction"
    ])

    # Function to download CSV from GitHub
    @st.cache_data
    def download_csv(url):
        response = requests.get(url)
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text))

    # Load dataset
    IC50_data_url = 'https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/CellLine_Dataset/Cleaned_PANCANCER_IC50.csv'
    df = download_csv(IC50_data_url)

    # Function to show statistical analysis and plots
    def show_statistics_and_plots(data, group_by_col):
        st.subheader(f'Statistical Analysis and Plots for {group_by_col}')
        
        # Statistical summary
        st.write(data.groupby(group_by_col).agg({
            'IC50': ['mean', 'std', 'min', 'max'],
            'AUC': ['mean', 'std', 'min', 'max'],
            'Max Conc': ['mean', 'std', 'min', 'max'],
            'RMSE': ['mean', 'std', 'min', 'max'],
            'Z score': ['mean', 'std', 'min', 'max']
        }))
        
        # Distribution plots
        for metric in ['IC50', 'AUC', 'Max Conc', 'RMSE', 'Z score']:
            st.subheader(f'Distribution of {metric} values by {group_by_col}')
            fig, ax = plt.subplots()
            sns.histplot(data[metric], bins=30, kde=True, ax=ax, color='orange')
            sns.kdeplot(data[metric], ax=ax, color='black')
            ax.set_xlabel(f'{metric} values')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
        
        # Pairplot
        st.subheader(f'Pairplot of Numerical Features by {group_by_col}')
        pairplot_fig = sns.pairplot(data, hue=group_by_col)
        st.pyplot(pairplot_fig)

    # Tab 1: Predictive Modeling
    with tab1:
        st.write("Predictive Modeling: Explore and predict drug sensitivity using Machine Learning Models:")
        
        # Sidebar filters
        st.write('Filters')
        tcga_class = st.multiselect('Select TCGA Classification (Optional):', df['TCGA Classification'].unique())
        tissue = st.multiselect('Select Tissue (Optional):', df['Tissue'].unique())
        drug = st.multiselect('Select Drug (Optional):', df['Drug Name'].unique())
        cell_line = st.multiselect('Select Cell Line (Optional):', df['Cell Line Name'].unique())
        
        # Apply filters
        filtered_df = df.copy()
        if tcga_class:
            filtered_df = filtered_df[filtered_df['TCGA Classification'].isin(tcga_class)]
        if tissue:
            filtered_df = filtered_df[filtered_df['Tissue'].isin(tissue)]
        if drug:
            filtered_df = filtered_df[filtered_df['Drug Name'].isin(drug)]
        if cell_line:
            filtered_df = filtered_df[filtered_df['Cell Line Name'].isin(cell_line)]
        
        # Label AUC values as '1' for sensitive and '0' for resistant
        filtered_df['Label'] = filtered_df['AUC'].apply(lambda x: 1 if x > 0.65 else 0)
        
        # Distribution of AUC values
        st.subheader('Distribution of AUC values')
        fig, ax = plt.subplots()
        sns.histplot(filtered_df['AUC'], bins=30, kde=True, ax=ax, color='orange')
        sns.kdeplot(filtered_df['AUC'], ax=ax, color='black')
        st.pyplot(fig)
        
        # Drug Sensitivity Distribution
        st.subheader('Drug Sensitivity Distribution')
        fig, ax = plt.subplots()
        sns.countplot(
            x='Label', 
            data=filtered_df, 
            ax=ax, 
            palette=['#FF8C00', '#1E90FF']
        )
        ax.set_xticklabels(['Resistant', 'Sensitive'])
        ax.set_xlabel('Drug Sensitivity')
        ax.set_ylabel('Count')
        ax.set_title('Drug Sensitivity Distribution', fontsize=14)
        st.pyplot(fig)
        
        # Tissue Distribution
        st.subheader('Tissue Distribution')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(y='Tissue', data=filtered_df, order=filtered_df['Tissue'].value_counts().index, ax=ax, palette='Set2')
        st.pyplot(fig)
        
        # Correlation heatmap of numerical features
        st.subheader('Correlation Heatmap of Numerical Features')
        corr = filtered_df[['IC50', 'AUC', 'Max Conc', 'RMSE', 'Z score']].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        
        # Pairplot of numerical features
        st.subheader('Pairplot of Numerical Features')
        pairplot_fig = sns.pairplot(filtered_df[['IC50', 'AUC', 'Max Conc', 'RMSE', 'Z score', 'Label']], hue='Label')
        st.pyplot(pairplot_fig)
        
        # Robust Models
        models = {
            'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(),
            'Support Vector Machine': SVC(probability=True),
            'Gradient Boosting': GradientBoostingClassifier(),
            'AdaBoost': AdaBoostClassifier(),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Decision Tree': DecisionTreeClassifier(),
            'Extra Trees': ExtraTreesClassifier(),
            'Bagging Classifier': BaggingClassifier()
        }
        
        # User to select model
        model_name = st.selectbox('Select Model', list(models.keys()))
        model = models[model_name]
        
        # Preparing data for model
        st.header('Model Training and Predictions')
        st.write(f'{model_name} model based on filtered data.')
        
        # Encoding categorical variables
        le = LabelEncoder()
        for col in ['Drug Name', 'Cell Line Name', 'TCGA Classification', 'Tissue', 'Tissue Sub-type']:
            filtered_df[col] = le.fit_transform(filtered_df[col])
        
        # Feature Extraction: Select top K best features
        X = filtered_df.drop(['AUC', 'Label', 'IC50', 'Max Conc', 'RMSE', 'Z score', 'Dataset Version'], axis=1)
        y = filtered_df['Label']
        
        # Scaling features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Feature selection using SelectKBest
        selector = SelectKBest(f_classif, k=10)
        X_new = selector.fit_transform(X_scaled, y)
        
        # Check if feature selection is successful
        if X_new.shape[1] == 0:
            st.error("No features selected by SelectKBest.")
            st.stop()
        
        # Ensure that y has the same number of samples as X_new
        if len(y) != X_new.shape[0]:
            st.error("Mismatch between the number of samples in X_new and y.")
            st.stop()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)
        
        # Train the selected model
        st.write(f'Training the {model_name} model based on filtered data.')
        model.fit(X_train, y_train)
        
        # Predicting
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Show metrics
        st.subheader('Model Performance')
        st.write('Accuracy:', accuracy_score(y_test, y_pred))
        st.write('ROC-AUC Score:', roc_auc_score(y_test, y_pred_proba))
        
        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        # Plotting the classification report as a heatmap
        st.subheader('Classification Report')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(report_df.iloc[:-1, :].T, annot=True, cmap='Blues', ax=ax)
        st.pyplot(fig)
        
        # ROC Curve
        st.subheader('ROC Curve')
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(fpr, tpr, color='#2a9d8f', lw=2, label=f'AUC = {roc_auc_score(y_test, y_pred_proba):.2f}')
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', label='Chance')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic Curve')
        ax.legend(loc='lower right')
        st.pyplot(fig)
        
    # Drug Analysis Tab
    with tab2:
        st.write("Drug Analysis: Analyze drug-specific data to explore cell lines and suggested tissues.")
            
        drug = st.selectbox('Select Drug:', df['Drug Name'].unique())
        drug_df = df[df['Drug Name'] == drug]
            
        # Show statistics
        show_statistics_and_plots(drug_df, 'Drug Name')
            
        # List of cell lines and suggested tissues
        st.subheader(f'Cell Lines and Suggested Tissues for {drug}')
        st.write(drug_df[['Cell Line Name', 'Tissue']].drop_duplicates().reset_index(drop=True))
        
    # Cell Line Analysis Tab
    with tab3:
        st.write("Cell Line Analysis: Analyze cell line-specific data to explore drugs and suggested tissues.")
            
        cell_line = st.selectbox('Select Cell Line:', df['Cell Line Name'].unique())
        cell_line_df = df[df['Cell Line Name'] == cell_line]
            
        # Show statistics
        show_statistics_and_plots(cell_line_df, 'Cell Line Name')
            
        # List of drugs and suggested tissues
        st.subheader(f'Drugs and Suggested Tissues for {cell_line}')
        st.write(cell_line_df[['Drug Name', 'Tissue']].drop_duplicates().reset_index(drop=True))
        
    # Tissue Analysis Tab
    with tab4:
        st.write("Tissue Analysis: Analyze tissue-specific data to explore cell lines and suggested drugs.")
            
        tissue = st.selectbox('Select Tissue:', df['Tissue'].unique())
        tissue_df = df[df['Tissue'] == tissue]
            
        # Show statistics
        show_statistics_and_plots(tissue_df, 'Tissue')
            
        # List of cell lines and suggested drugs
        st.subheader(f'Cell Lines and Suggested Drugs for {tissue}')
        st.write(tissue_df[['Cell Line Name', 'Drug Name']].drop_duplicates().reset_index(drop=True))
        
    # Comprehensive Analysis Tab
    with tab5:
        st.write("Comprehensive Analysis: Perform comprehensive exploratory data analysis and statistical analysis.")
            
        st.subheader('Distribution of IC50 Values by Tissue')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='Tissue', y='IC50', data=df, ax=ax, palette='Set2')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig)
        
        st.subheader('IC50 by Drug')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='Drug Name', y='IC50', data=df, ax=ax, palette='Set2')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig)
        
        # Comprehensive statistical analysis
        st.subheader('Comprehensive Statistical Analysis')
        st.write(df[['IC50', 'AUC', 'Max Conc', 'RMSE', 'Z score']].describe())
        
        # Correlation heatmap of numerical features
        st.subheader('Correlation Heatmap of Numerical Features')
        corr = df[['IC50', 'AUC', 'Max Conc', 'RMSE', 'Z score']].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        
    # In-Vitro Prediction Tab
    # Define functions to make predictions based on user input
    # Function to preprocess new input data
    def preprocess_input(input_value, feature_names):
        # Dummy values based on feature names to ensure the correct size and format
        input_features = pd.DataFrame([0] * len(feature_names)).T
        input_features.columns = feature_names
        
        # Example preprocessing step - customize this according to your feature requirements
        # Here we use 0 for numerical features and encode categorical as unknown category
        for feature in feature_names:
            # Set default or encoded values based on feature type
            input_features[feature] = 0
            
        return input_features
        
    # Prediction function for both Drugs and Cell Lines
    def predict_any_input(input_value, input_type, model, df, feature_names):
        # Case-insensitive input handling
        input_value = input_value.strip().upper()
        
        # Convert relevant columns to string and handle missing values
        if input_type == 'Drug':
            df['Drug Name'] = df['Drug Name'].astype(str).fillna('')
            matching_df = df[df['Drug Name'].str.upper() == input_value]
        else:
            df['Cell Line Name'] = df['Cell Line Name'].astype(str).fillna('')
            matching_df = df[df['Cell Line Name'].str.upper() == input_value]
            
        if matching_df.empty:
            # Handle new or unseen inputs
            #st.info(f"No exact data available for {input_type}: {input_value}. Predicting based on general input.")
            input_features = preprocess_input(input_value, feature_names)
        else:
            # If matching data is found in the CSV
            input_features = matching_df.drop(['AUC', 'Label', 'IC50', 'Max Conc', 'RMSE', 'Z score', 'Dataset Version'], axis=1, errors='ignore')
            input_features = input_features.iloc[0:1]  # Take the first row as sample input
        
        # Preprocess features
        X_scaled = scaler.transform(input_features)
        X_new = selector.transform(X_scaled)
        prediction = model.predict(X_new.mean(axis=0).reshape(1, -1))
        return 'Sensitive' if prediction[0] == 1 else 'Resistant'
        
    # Tab 6: Options for Prediction
    with tab6:
        st.write("Enter a Drug or Cell Line to get predictions.")
        
        # Load dataset for feature extraction and model
        df = download_csv(IC50_data_url)
            
        # Label AUC values as '1' for sensitive and '0' for resistant
        df['Label'] = df['AUC'].apply(lambda x: 1 if x > 0.65 else 0)
            
        # Preprocessing for feature extraction
        le = LabelEncoder()
        for col in ['Drug Name', 'Cell Line Name', 'TCGA Classification', 'Tissue', 'Tissue Sub-type']:
            if col in df.columns:
                df[col] = le.fit_transform(df[col])
            
        # Feature Extraction
        X = df.drop(['AUC', 'Label', 'IC50', 'Max Conc', 'RMSE', 'Z score', 'Dataset Version'], axis=1, errors='ignore')
        y = df['Label']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        selector = SelectKBest(f_classif, k=10)
        X_new = selector.fit_transform(X_scaled, y)
            
        # Train XGBoost model
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        
        # Extract feature names for prediction inputs
        feature_names = X.columns
        
        # User input
        input_type = st.selectbox('Select Input Type:', ['Drug', 'Cell Line'])
        input_value = st.text_input(f'Enter the {input_type}:')
        
        if st.button('Predict'):
            if input_value:
                prediction = predict_any_input(input_value, input_type, model, df, feature_names)
                st.write(f'Prediction for {input_type} "{input_value}": {prediction}')
            else:
                st.error("Please enter a valid Drug or Cell Line.")

        st.write('Thank you for using the Tissue and Cell Line-specific Essential Drug Prediction Tool!')    

# Tool 13: NetPharm
elif selected_tool == "Tool 13: NetPharm":
    st.markdown("<h4 style='color: #3498db;'>Tool 13: NetPharm</h4>", unsafe_allow_html=True)
    st.markdown("""
Demonstrate interactions to decide what network are potential targets for therapeutic intervention.
""")
    
    st.markdown("---")
    # Main input section
    gene_input = st.text_area("Enter Gene:", "TP53", height=100)  # Adjust height as needed
    gene_list = [gene.strip() for gene in gene_input.split(',')]

    if st.button("Run Analysis"):
        # Fetch data
        gene_interactions = fetch_gene_interactions(gene_list)
        drug_target_df = fetch_drug_target_interactions(gene_list)
        enrichment_df = fetch_gprofiler_enrichment_results(gene_list)
        
        if not gene_interactions:
            st.write("No gene interactions found.")
        else:
            # Create network graph
            G = nx.Graph()
            for edge in gene_interactions:
                G.add_edge(edge[0], edge[1], weight=edge[2])
            
            # Animated PPI plot
            fig_animated = plot_animated_network(G)
            st.plotly_chart(fig_animated, use_container_width=True)
        
        # Display drug-target interaction table
        if not drug_target_df.empty:
            st.write("Drug-Target Interactions:")
            st.dataframe(drug_target_df)
            
            # Drug-Target Plot
            fig_drug_target = px.scatter(drug_target_df, x="ZINC id", y="pKi", color="Purchasability", size="pKi", hover_name="ZINC id", title="Drug-Target Interactions")
            st.plotly_chart(fig_drug_target)
            
            # Venn Plot for Overlapping Targets and Drugs
            targets = set(gene_list)
            drugs = set(drug_target_df["Smiles"].unique())  
            plot_venn_diagram(targets, drugs)

            # Statistical Plot for Degree Distribution
            if not drug_target_df.empty:
                st.write("Degree Distribution of Protein-Protein Interaction Network:")
                
                # Calculate degree for each node
                degree_sequence = [(node, val) for node, val in G.degree()]
                
                # Create DataFrame
                degree_df = pd.DataFrame(degree_sequence, columns=["Node", "Degree"])
                
                # Plot Degree Distribution
                fig_degree = px.histogram(degree_df, x="Degree", title="Degree Distribution of Nodes")
                st.plotly_chart(fig_degree)
            
        else:
            st.write("No drug-target interactions found.")
        
        # Display Enrichment Results
        if enrichment_df is not None:
            st.write("Enrichment Analysis Results:")
            st.dataframe(enrichment_df)  
            
            # Plot the enrichment results
            plot_enrichment_results(enrichment_df)
        else:
            st.write("No enrichment results found.")

elif selected_tool == "Tool 14: QSARGx":
    st.markdown("<h4 style='color: #3498db;'>Tool 14: QSARGx</h4>", unsafe_allow_html=True)
    st.markdown("""
QSAR is a mathematical model that predicts a chemical's biological activity.
""")
    
    st.markdown("---")
    file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if file is not None:
        df = load_data(file)
        if df is not None:
            # Normalize column names
            df.columns = df.columns.str.strip().str.lower()
            
            # Check if the required columns exist
            if 'smiles' in df.columns and 'pic50' in df.columns:
                compound_df = df[['smiles', 'pic50']]
                st.write("Data Preview:", compound_df.head())
                
                if st.checkbox("Generate Fingerprints and Train Model"):
                    compounds, fingerprints = generate_fingerprints(compound_df)

                    if len(fingerprints) == 0:
                        st.error("Failed to generate fingerprints. Please check the SMILES strings.")
                    else:
                        labels = compound_df['pic50'].values

                        # Prepare data for deep learning
                        X, y = prepare_data_for_model(fingerprints, labels)

                        # Generate indices for train-test split
                        indices = np.arange(len(labels))
                        X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
                            X, y, indices, test_size=0.2, random_state=42
                        )

                        # Train and evaluate the model
                        model = train_model(X_train, X_test, y_train, y_test, compound_df, test_indices)
                        
                        st.success("Model trained successfully!")
            else:
                st.error("The required columns 'smiles' and 'pic50' are not present in the dataset.")
        else:
            st.error("Failed to load data. Please check the file format.")
    else:
        st.warning("Please upload a CSV file.")


elif selected_tool == "Tool 15: ChemSyn":
    st.markdown("<h4 style='color: #3498db;'>Tool 15: ChemSyn</h4>", unsafe_allow_html=True)
    st.markdown("""
    ChemSyn: Chemical Reaction Transformation & Molecular Analysis.
    """)
    st.markdown("---")
    # Session state to handle page loading
    if 'page_loaded' not in st.session_state:
        st.session_state.page_loaded = False
    
    # Tab options (Ensure correct indentation)
    tab1, tab2 = st.tabs(["Chemical Transformations Analysis", "Chemical Reactions Analysis"])

    # Tab 1 Script (Chemical Transformations & Molecular Analysis)
    with tab1:
        st.header("Upload Files")
        smiles_file = st.file_uploader("Upload SMILES file (txt)", type=["txt"])
        reactions_file = st.file_uploader("Upload Reactions file (txt)", type=["txt"])
        max_cycles = st.slider("Max Reaction Cycles", min_value=1, max_value=20, value=10)
    
        if smiles_file and reactions_file:
            smiles_data = smiles_file.read().decode("utf-8").strip().split("\n")
            reactions_data = reactions_file.read().decode("utf-8").strip().split("\n")
    
            if st.button("Apply Transformations", key="apply_transformations"):
                st.session_state.page_loaded = True
                results = []
                for smile in smiles_data:
                    mol = Chem.MolFromSmiles(smile)
                    if mol is None:
                        results.append({"SMILES": smile, "Result": "Invalid SMILES"})
                        continue
    
                    transformed = False
                    for rxn_smarts in reactions_data:
                        try:
                            reaction = rdChemReactions.ReactionFromSmarts(rxn_smarts)
                            if reaction is None:
                                results.append({"SMILES": smile, "Result": "Invalid Reaction SMARTS"})
                                continue
    
                            products = reaction.RunReactants([mol])
                            if not products:
                                continue
    
                            product_smiles = [Chem.MolToSmiles(prod) for prod in products[0]]
                            results.append({"SMILES": smile, "Result": ", ".join(product_smiles)})
                            transformed = True
                            break
                        except Exception as e:
                            results.append({"SMILES": smile, "Result": f"Error: {e}"})
    
                    if not transformed:
                        results.append({"SMILES": smile, "Result": "No products"})
    
                df = pd.DataFrame(results)
                st.subheader("Transformation Results")
                st.write(df)
    
                # Download as CSV
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name='transformation_results.csv',
                    mime='text/csv',
                    key="csv_download"
                )
    
                # Option to download as SDF
                sdf_buffer = StringIO()
                writer = Chem.SDWriter(sdf_buffer)
                for smiles, name in zip(smiles_data, [f"Molecule_{i+1}" for i in range(len(smiles_data))]):
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        mol.SetProp("_Name", name)
                        writer.write(mol)
                writer.close()
                sdf_data = sdf_buffer.getvalue().encode('utf-8')
                st.download_button(
                    label="Download Results as SDF",
                    data=sdf_data,
                    file_name='transformation_results.sdf',
                    mime='chemical/x-mdl-sdfile',
                    key="sdf_download"
                )
    
                # Molecular Property Analysis
                st.subheader("Molecular Property Analysis")
                analysis_results = []
                for smiles in [result["Result"] for result in results if result["Result"] != "No products"]:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        continue
                    mw = Descriptors.MolWt(mol)
                    logp = Descriptors.MolLogP(mol)
                    tpsa = Descriptors.TPSA(mol)
                    rot_bonds = Descriptors.NumRotatableBonds(mol)
                    h_donors = Descriptors.NumHDonors(mol)
                    h_acceptors = Descriptors.NumHAcceptors(mol)
                    analysis_results.append({
                        'SMILES': smiles, 
                        'MW': mw, 
                        'LogP': logp, 
                        'TPSA': tpsa, 
                        'Rotatable Bonds': rot_bonds, 
                        'H-Bond Donors': h_donors, 
                        'H-Bond Acceptors': h_acceptors
                    })
                analysis_df = pd.DataFrame(analysis_results)
                st.write(analysis_df)
    
                # Visualize Molecular Properties
                st.subheader("Molecular Properties Distribution")
                if not analysis_df.empty:
                    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
                    ax[0, 0].hist(analysis_df['MW'], bins=20, color='blue', alpha=0.7, edgecolor='black')
                    ax[0, 0].set_title("Molecular Weight Distribution", fontsize=14)
                    ax[0, 0].set_xlabel("Molecular Weight (MW)", fontsize=12)
                    ax[0, 0].set_ylabel("Frequency", fontsize=12)
                    ax[0, 1].hist(analysis_df['LogP'], bins=20, color='green', alpha=0.7, edgecolor='black')
                    ax[0, 1].set_title("LogP Distribution", fontsize=14)
                    ax[0, 1].set_xlabel("LogP", fontsize=12)
                    ax[0, 1].set_ylabel("Frequency", fontsize=12)
                    ax[1, 0].hist(analysis_df['TPSA'], bins=20, color='red', alpha=0.7, edgecolor='black')
                    ax[1, 0].set_title("TPSA Distribution", fontsize=14)
                    ax[1, 0].set_xlabel("Topological Polar Surface Area (TPSA)", fontsize=12)
                    ax[1, 0].set_ylabel("Frequency", fontsize=12)
                    ax[1, 1].scatter(analysis_df['MW'], analysis_df['LogP'], color='purple', alpha=0.7, edgecolor='black')
                    ax[1, 1].set_title("MW vs LogP", fontsize=14)
                    ax[1, 1].set_xlabel("Molecular Weight (MW)", fontsize=12)
                    ax[1, 1].set_ylabel("LogP", fontsize=12)
                    plt.tight_layout(pad=3.0)
                    st.pyplot(fig)
    
        st.subheader("Example SMILES and Reactions")
        st.text("Example SMILES: \nc1ccccc1\nC1CCCCC1")
        st.text("Example Reactions: \nc1ccccc1>>Brc1ccccc1 (Bromination)\nC1CCCCC1>>C1CCCOCC1 (Oxidation)")
        
        st.write("""**About ChemSyn**""")
        st.markdown("""
        **Tool Description:**
        This tool enables users to apply various chemical transformations to SMILES strings, analyze molecular properties, and visualize distributions and comparisons of different molecular properties.
        """)
    
    # Tab 2 Script (Descriptor Computation & Visualization)
    with tab2:
        input_smiles = st.text_input('Enter SMILES string:')
        if input_smiles:
            pattern = re.compile(r'[^CcNnOoPpSsFfClBrIiHh0-9@+\-=#()%[\]]')
            if pattern.search(input_smiles):
                st.error("Invalid SMILES string.")
                st.stop()
    
        transformation = st.selectbox('Choose a chemical transformation:', [
            'Sulfonamidation', 'Alkylation', 'Amination', 'Aminolysis', 'Acylation', 
            'Carbamoylation', 'Oxidation', 'Reduction', 'Aryl Coupling', 'Esterification',
            'Etherification', 'Nitration', 'Halogenation', 'Cyclization', 'Dehydration',
            'Deprotection', 'Methylation', 'Phosphorylation', 'Hydrogenation', 'Hydrolysis',
            'Transesterification', 'Condensation', 'Hydroamination', 'Hydroformylation',
            'Ring-Closing Metathesis (RCM)', 'Grignard Reaction', 'Friedel-Crafts Alkylation/Acyloylation',
            'Michael Addition', 'Diels-Alder Reaction', 'Cyclopropanation', 'Wittig Reaction',
            'Claisen Condensation', 'Buchwald-Hartwig Coupling', 'Sonogashira Coupling',
            'Suzuki-Miyaura Coupling', 'Stille Coupling', 'Negishi Coupling', 'Silylation',
            'Tosylation', 'Mesylation', 'Fischer Esterification', 'Vilsmeier-Haack Reaction',
            'Pfitzner-Moffatt Oxidation', 'Mitsunobu Reaction', 'Borylation'
        ])
    
        if st.button('Apply Reactions', key="apply_reactions"):
            st.session_state.page_loaded = True
            transformations = {
                'Sulfonamidation': input_smiles.replace('c', 'cS(=O)(=O)N'),
                'Alkylation': input_smiles + 'C',
                'Amination': input_smiles + 'N',
                'Aminolysis': input_smiles.replace('C(=O)', 'C(=O)N'),
                'Acylation': input_smiles + 'C(=O)R',
                'Carbamoylation': input_smiles + 'C(=O)N',
                'Oxidation': input_smiles.replace('C', 'CO'),
                'Reduction': input_smiles.replace('O', 'OH'),
                'Aryl Coupling': input_smiles + 'C',
                'Esterification': input_smiles + 'OCO',
                'Etherification': input_smiles + 'O',
                'Nitration': input_smiles + 'NO2',
                'Halogenation': input_smiles + 'Cl',
                'Cyclization': input_smiles + 'C1CCCCC1',
                'Dehydration': input_smiles.replace('OH', ''),
                'Deprotection': input_smiles.replace('C(O)', 'C'),
                'Methylation': input_smiles + 'C',
                'Phosphorylation': input_smiles + 'PO4',
                'Hydrogenation': input_smiles + 'H2',
                'Hydrolysis': input_smiles.replace('COOH', 'COH'),
                'Transesterification': input_smiles.replace('COO', 'COOCH3'),
                'Condensation': input_smiles.replace('OH', 'O'),
                'Hydroamination': input_smiles + 'N',
                'Hydroformylation': input_smiles + 'CHO',
                'Ring-Closing Metathesis (RCM)': input_smiles.replace('C=C', 'C1CC1'),
                'Grignard Reaction': input_smiles + 'MgBr',
                'Friedel-Crafts Alkylation/Acyloylation': input_smiles + 'C(=O)R',
                'Michael Addition': input_smiles + 'CC=C',
                'Diels-Alder Reaction': input_smiles + 'C=C-C=C',
                'Cyclopropanation': input_smiles + 'C1C(C1)C',
                'Wittig Reaction': input_smiles + 'C=C',
                'Claisen Condensation': input_smiles + 'C(=O)C',
                'Buchwald-Hartwig Coupling': input_smiles + 'C',
                'Sonogashira Coupling': input_smiles + 'C#C',
                'Suzuki-Miyaura Coupling': input_smiles + 'C',
                'Stille Coupling': input_smiles + 'C',
                'Negishi Coupling': input_smiles + 'C',
                'Silylation': input_smiles + 'Si',
                'Tosylation': input_smiles + 'C6H4SO3',
                'Mesylation': input_smiles + 'CH3SO2',
                'Fischer Esterification': input_smiles + 'OCOC',
                'Vilsmeier-Haack Reaction': input_smiles + 'C=O',
                'Pfitzner-Moffatt Oxidation': input_smiles + 'O',
                'Mitsunobu Reaction': input_smiles + 'C=N',
                'Borylation': input_smiles + 'B'
            }
    
            transformed_smiles = transformations.get(transformation, input_smiles)
            #st.subheader("Transformed SMILES")
            #st.text(transformed_smiles)
    
            # Descriptor Computation
            mol = Chem.MolFromSmiles(transformed_smiles)
            if mol:
                descriptors = {
                    'Molecular Weight': Descriptors.MolWt(mol),
                    'LogP': Descriptors.MolLogP(mol),
                    'TPSA': Descriptors.TPSA(mol),
                    'Rotatable Bonds': Descriptors.NumRotatableBonds(mol),
                    'H-Bond Donors': Descriptors.NumHDonors(mol),
                    'H-Bond Acceptors': Descriptors.NumHAcceptors(mol)
                }
                
                #st.write(descriptors)
                
                # Evaluate Lipinski's Rule of Five
                lipinski = lipinski_rule(descriptors)
                st.write(f'Lipinski\'s Rule of Five: {"Pass" if lipinski else "Fail"}')
                
                # Molecular Property Comparison Plot
                #st.text('Molecular Property Comparison')
                descriptor_data = []
                for trans in generate_transformation_options():
                    transformed_smiles = perform_transformation(input_smiles, trans)
                    descriptors_after = compute_molecular_descriptors(transformed_smiles)
                    sa_score = compute_sa_score(Chem.MolFromSmiles(transformed_smiles))
                    descriptor_data.append({
                        'Transformation': trans,
                        'Molecular Weight': descriptors_after.get('Molecular Weight', np.nan),
                        'LogP': descriptors_after.get('LogP', np.nan),
                        'TPSA': descriptors_after.get('TPSA', np.nan),
                        'SA Score': sa_score
                    })
                
                descriptor_df = pd.DataFrame(descriptor_data)
                
                # Plotting Descriptor Comparison Across Transformations
                fig, ax = plt.subplots(figsize=(12, 8))  # Increase figure size for better clarity
                descriptor_df.set_index('Transformation').plot(kind='bar', ax=ax, width=0.8)  # Adjust bar width
                
                # Set plot title and labels
                ax.set_title('Descriptor Comparison Across Transformations', fontsize=16)
                ax.set_xlabel('Transformation', fontsize=14)
                ax.set_ylabel('Descriptor Values', fontsize=14)
                
                # Rotate x-axis labels for clarity and add padding to separate legends
                plt.xticks(rotation=45, ha='right', fontsize=12)
                plt.tight_layout()  # Automatically adjust subplot params for better spacing
                
                # Show plot in Streamlit
                st.pyplot(fig)
            
                # Molecular Weight Distribution Plot
                #st.text('Molecular Weight Distribution')
                weights = [descriptors['Molecular Weight']]
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(weights, bins=20, color='skyblue', edgecolor='black')
                ax.set_title('Molecular Weight Distribution')
                ax.set_xlabel('Molecular Weight')
                ax.set_ylabel('Frequency')
                st.pyplot(fig)
            
                # Pairwise Descriptor Correlation Heatmap
                #st.header('Descriptor Correlation Heatmap')
                #descriptor_df = pd.DataFrame([descriptors])
                #corr = descriptor_df.corr()
                #fig, ax = plt.subplots(figsize=(10, 6))
                #sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
                #ax.set_title('Descriptor Correlation Heatmap')
                #st.pyplot(fig)
            
                # LogP vs. Molecular Weight Scatter Plot
                #st.header('LogP vs. Molecular Weight')
                #fig, ax = plt.subplots(figsize=(10, 6))
                #ax.scatter(descriptors['Molecular Weight'], descriptors['LogP'], color='coral')
                #ax.set_title('LogP vs. Molecular Weight')
                #ax.set_xlabel('Molecular Weight')
                #ax.set_ylabel('LogP')
                #st.pyplot(fig)
            
                # Summary Table for Molecular Properties
                st.text('Molecular Properties Summary')
                summary_df = pd.DataFrame({
                    'Property': ['Molecular Weight', 'LogP', 'TPSA', 'SA Score'],
                    'Value': [
                        descriptors.get('Molecular Weight', np.nan),
                        descriptors.get('LogP', np.nan),
                        descriptors.get('TPSA', np.nan),
                        descriptors.get('SA Score', np.nan)
                    ]
                })
                st.write(summary_df)
                
                # Download SDF File
                sdf_data = smiles_to_sdf([transformed_smiles], ["Transformed Molecule"])
                st.download_button("Download SDF", sdf_data, "transformed_molecule.sdf", "chemical/x-mdl-sdfile")
    
    
        st.subheader("Example SMILES")
        st.text("Example SMILES: \nCCOCC\nCCOC(=O)C")
        
        st.write("""**About ChemSyn**""")
        st.markdown("""
        **Tool Description:**
        This section of the tool allows users to perform various chemical transformations on SMILES strings, compute molecular descriptors such as molecular weight, logP, and TPSA, and visualize the results.
        """)

# Tool 16: ChemPROTAC
elif selected_tool == "Tool 16: ChemPROTAC":
    st.markdown("<h4 style='color: #3498db;'>Tool 16: Deep Learning Driven Server for PROTAC Prediction</h4>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Load dataset
    df = load_data16()
    df.fillna(0, inplace=True)
    label_encoder = LabelEncoder()
    df['mol_encoded'] = label_encoder.fit_transform(df['mol'])
    df['mol_fp'] = df['mol'].apply(mol_to_fp16)

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Model Prediction", "Similarity Search",  "Generative Modelling"])
    
    # Tab 1: Model Prediction
    with tab1:
        st.subheader("Model Prediction")
        
        # Data preprocessing for model training
        X = np.array(df['mol_fp'].to_list())
        y = df['flag']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    
        # Reshape for LSTM input
        X_train_scaled_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
        X_test_scaled_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
        
        # Update the models dictionary
        models = {
            "NeuralNet": Sequential([Dense(1024, activation='relu'), Dense(512, activation='relu'), Dense(1, activation='sigmoid')]),
            "LSTM": Sequential([LSTM(128, input_shape=(1, X_train_scaled.shape[1])), Dense(1, activation='sigmoid')]),
            "SimpleRNN": Sequential([SimpleRNN(128, input_shape=(1, X_train_scaled.shape[1])), Dense(1, activation='sigmoid')])
        }
        
        # Train models with the correct data shapes

        trained_models = {}
        for name, model in models.items():
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            if name in ["LSTM", "SimpleRNN"]:
                model.fit(X_train_scaled_lstm, y_train, epochs=10, batch_size=32)
            else:
                model.fit(X_train_scaled, y_train, epochs=10, batch_size=32)
            trained_models[name] = model
    
        user_smiles = st.text_input("Enter a SMILES string:", value="CCN(CCCC#CC1=C[N](CCC(=O)N[C@H](C(=O)N2C[C@H](O)C[C@H]2C(=O)N[C@@H](C)C3=CC=C(C=C3)C4=C(C)N=CS4)C(C)(C)C)N=C1)CCOC5=CC=C(C=C5)C(=O)C6=C(SC7=CC(=CC=C67)O)C8=CC=C(O)C=C8")
        
        if st.button("Predict"):
            if user_smiles:
                predictions = predict_protac_activity(trained_models, scaler, user_smiles)
                st.subheader("Molecular Properties:")
                properties = extract_molecular_properties(user_smiles)
                for prop, value in properties.items():
                    st.write(f"{prop}: {value}")
    
                st.subheader("Predictions:")
                for name, prob in predictions.items():
                    activity = describe_activity(prob)
                    st.write(f"{name}: {prob:.4f} ({activity})")
    
                df_pred = pd.DataFrame(predictions.items(), columns=['Model', 'Predicted Probability'])
                st.write(df_pred)
    
                st.subheader("Predicted Activity Probability:")
                activity_prob_plot = plot_activity_probability(predictions["NeuralNet"])
                st.pyplot(activity_prob_plot)
    
                st.subheader("AUC-ROC Curves:")
                auc_plot = plot_roc_auc_curve(trained_models, X_test_scaled, y_test)
                st.pyplot(auc_plot)
                
                st.markdown(get_binary_file_downloader_html(df_pred, file_name="predictions.csv"), unsafe_allow_html=True)
    
        # Model Accuracy Comparison
        if st.button("Compare Models"):
            performance = compare_model_performance(trained_models, X_test_scaled, y_test)
        
            st.subheader("Model Accuracy Comparison")
        
            fig, ax = plt.subplots(figsize=(10, 5))  # Adjust the figure size
            ax.bar(performance.keys(), performance.values(), color='purple')
            ax.set_xlabel('Models/Tools')
            ax.set_ylabel('Accuracy')
            ax.set_title('Model Accuracy vs Public Tools')
            ax.tick_params(axis='x', rotation=30)  # Adjust rotation
            st.pyplot(fig)


    # Tab 2: Similarity Search
    with tab2:
        st.subheader("Molecular Similarity Search")
        
        query_smiles = st.text_input("Enter a SMILES string for similarity search:", value="CCN(CCCC#CC1=C[N](CCC(=O)N[C@H](C(=O)N2C[C@H](O)C[C@H]2C(=O)N[C@@H](C)C3=CC=C(C=C3)C4=C(C)N=CS4)C(C)(C)C)N=C1)CCOC5=CC=C(C=C5)C(=O)C6=C(SC7=CC(=CC=C67)O)C8=CC=C(O)C=C8")
        
        if st.button("Search Similarity"):
            if query_smiles:
                query_fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(query_smiles), 2, nBits=1024)
                
                # Calculate Tanimoto similarity
                similarities = []
                for index, row in df.iterrows():
                    ref_fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(row['mol']), 2, nBits=1024)
                    similarity = DataStructs.TanimotoSimilarity(query_fp, ref_fp)
                    similarities.append((row['mol'], similarity))
                
                # Sort and display results
                similarities.sort(key=lambda x: x[1], reverse=True)
                results_df = pd.DataFrame(similarities, columns=["SMILES", "Tanimoto Similarity"])
                st.write(results_df.head(10))  # Display top 10 similar molecules
    
                # Plot similarity distribution
                plt.figure(figsize=(10, 5))
                plt.hist([sim[1] for sim in similarities], bins=20, color='skyblue', edgecolor='black')
                plt.xlabel('Tanimoto Similarity')
                plt.ylabel('Frequency')
                plt.title('Similarity Distribution')
                st.pyplot(plt)
    
                # Plot top similar compounds
                top_similar = results_df.head(10)
                plt.figure(figsize=(10, 5))
                plt.bar(top_similar['SMILES'], top_similar['Tanimoto Similarity'], color='purple')
                plt.xticks(rotation=45, ha='right')
                plt.xlabel('Similar Compounds (SMILES)')
                plt.ylabel('Tanimoto Similarity')
                plt.title('Top 10 Similar Compounds')
                st.pyplot(plt)
    
                # Generate SDF file for download
                sdf_data = smiles_to_sdf(top_similar['SMILES'], [f"Compound {i}" for i in range(10)])
                st.download_button("Download SDF", sdf_data, "top_similar_protac.sdf", "chemical/x-mdl-sdfile")


    # Generative Modelling tab
    # Tab 3: Generative Modelling
    
    # Generative Modelling tab
    with tab3:
        st.subheader("Generative Modelling")
        
        seed_smiles = st.text_input("Enter Seed SMILES:", value="CCN(CCCC#CC1=C[N](CCC(=O)N[C@H](C(=O)N2C[C@H](O)C[C@H]2C(=O)N[C@@H](C)C3=CC=C(C=C3)C4=C(C)N=CS4)C(C)(C)C)N=C1)CCOC5=CC=C(C=C5)C(=O)C6=C(SC7=CC(=CC=C67)O)C8=CC=C(O)C=C8")
    
        if st.button("Generate"):
            if seed_smiles:
                # Convert seed SMILES to a molecular fingerprint and find similar compounds
                seed_fp = mol_to_fp16(seed_smiles)
                neighbor_model = NearestNeighbors(n_neighbors=10)
                neighbor_model.fit(np.array(df['mol_fp'].tolist()))
    
                distances, indices = neighbor_model.kneighbors([seed_fp])
                similar_smiles = df.iloc[indices[0]]['mol'].tolist()
    
                # Display ROC-AUC Curve
                st.write("ROC-AUC Curve:")
                st.pyplot(plot_roc_auc_curve(models, X_test, y_test))
    
                # Display top 10 generated PROTAC SMILES
                if len(similar_smiles) > 0:
                    st.write("### Top 10 Generated PROTAC SMILES")
                    smiles_data = [{"Index": i, "SMILES": smi} for i, smi in enumerate(similar_smiles[:10])]
                    smiles_df = pd.DataFrame(smiles_data)
                    st.table(smiles_df)
    
                    # Calculate toxicity features for generated SMILES
                    toxicity_results = [calculate_toxicity_features(smi) for smi in similar_smiles[:10]]
                    toxicity_df = pd.DataFrame(toxicity_results)
    
                    # Display toxicity features
                    st.write("### Toxicity Features")
                    st.table(toxicity_df)
    
                    # Plot toxicity features
                    toxicity_df[['TSP', 'LogP', 'Molecular Weight', 'Aromatic Rings']].plot(kind='box', title='Toxicity Feature Distribution')
                    plt.xticks(rotation=30)  # Rotate labels for better visibility
                    plt.tight_layout()  # Adjust layout to prevent overlap
                    st.pyplot(plt)

                    # Plot toxicity features
                    st.bar_chart(toxicity_df.set_index('SMILES')[['TSP', 'LogP', 'Molecular Weight', 'Aromatic Rings']])
    
                    # Generate SDF file for download
                    sdf_data = smiles_to_sdf(similar_smiles[:10], [f"Compound {i}" for i in range(10)])
                    st.download_button("Download SDF", sdf_data, "generated_protac.sdf", "chemical/x-mdl-sdfile")
    
                else:
                    st.write("No similar SMILES generated.")
                    st.markdown("---")
                    
    st.markdown("---")
    st.write("""**About ChemPROTAC**""")
    st.write("Model Prediction Tab: In this tab, users can input a SMILES string to predict PROTAC activity using deep learning models. The tool provides molecular properties, predicted probabilities, and visualizations of activity probability and model performance. It allows users to download predictions as a CSV file for further analysis.")
    st.write("Similarity Search Tab: Users can enter a SMILES string to perform a similarity search against a dataset. This tab computes Tanimoto similarity scores and displays the top similar compounds. It also visualizes the similarity distribution and allows users to download results in SDF format.")
    st.write("Generative Modelling Tab: This section enables users to generate new PROTAC SMILES based on a seed input. It showcases the top 10 similar compounds and their toxicity features, including LogP and molecular weight. Users can visualize the toxicity analysis and download the generated compounds as an SDF file.")                

# Tool 17: NanoMedScore: Nanomedicine Prediction
elif selected_tool == "Tool 17: NanoMedScore":
    st.markdown("<h4 style='color: #3498db;'>Tool 17: NanoMedScore: Nanomedicine Prediction</h4>", unsafe_allow_html=True)
    
    st.markdown("---")
    # Load and display data
    data = load_data17()
    
    # Preprocess data
    features, target = preprocess_data17(data)
    
    # API selection dropdown
    st.title("API Selection")
    api_list = data['API'].unique().tolist()
    selected_api = st.selectbox("Select API", api_list)
    
    # Filter data based on selected API
    filtered_data = data[data['API'] == selected_api]
    
    # Model Selection
    st.title("Model Selection")
    model_type = st.selectbox("Choose a Machine Learning Model", ("XGBoost", "Random Forest", "SVM", "MLP"))
    
    # Train model
    model, accuracy, X_test, y_test, predictions = train_model17(features, target, model_type)
    
    # Display model performance
    st.write(f"### {model_type} Model Accuracy: {accuracy * 100:.2f}%")
    
    # Visualization options
    st.subheader("Visualizations")
    
    # Confusion matrix
    #if st.checkbox("Show Confusion Matrix"):
        #plot_confusion_matrix(y_test, predictions)
    
    # Feature importance
    if st.checkbox("Show Feature Importance"):
        plot_feature_importance17(model, features, model_type)
    
    # ROC curve
    if st.checkbox("Show ROC Curve"):
        plot_roc_curve17(model, X_test, y_test)
    
    # Advanced Nanomedicine Analysis Section
    st.title("Advanced Analysis")
    analysis_type = st.selectbox("Choose an Analysis", ("EDA", "Pharmacokinetics", "Physicochemical Properties"))
    
    # Exploratory Data Analysis (EDA) - Correlation Matrix
    if analysis_type == "EDA":
        eda_analysis(filtered_data)
    
    # Nanoparticle Analysis
    if st.checkbox("Nanoparticle Analysis"):
        nanoparticle_analysis(filtered_data)
    
    # Pharmacokinetics analysis
    if analysis_type == "Pharmacokinetics":
        pharmacokinetics_analysis(filtered_data)
        # Note for Routes
        st.write("""
        **Note**: 
        - **IV (Intravenous)**: Administered directly into the bloodstream.
        - **PO (Oral)**: Taken by mouth and absorbed through the digestive system.
        - **IP (Intraperitoneal)**: Injected into the abdominal cavity.
        - **IN (Intranasal)**: Administered through the nasal passages for rapid absorption.
        """)
    
    # Physicochemical Properties Analysis
    elif analysis_type == "Physicochemical Properties":
        st.write("### Physicochemical Properties Analysis")
        st.write("Size Distribution")
        fig = px.histogram(data, x='Size', nbins=20, title="Size Distribution", template="plotly_dark")
        st.plotly_chart(fig)
        st.markdown("---")
    
    st.markdown("---")
    st.write("""**About NanoMedScore**""")
    st.write("NanoMedScore allows users to predict the pharmacokinetic properties of nanoparticles using machine learning models. By leveraging various algorithms such as XGBoost and Random Forest, users can analyze features like molecular weight, solubility, and drug-carrier ratios. The tool also includes visualizations like confusion matrices, ROC curves, and exploratory data analysis (EDA), enabling researchers to gain insights into the behavior and efficacy of nanomedicine formulations.")


# Tool 18: DrugSwitch: The process of finding new uses for existing drugs
elif selected_tool == "Tool 18: DrugSwitch":
    st.markdown("<h4 style='color: #3498db;'>Tool 18: DrugSwitch: Multi-Omics Drug-Disease Relationship Pred</h4>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # User Input
    disease, drug, metabolite, mirna, mrna, protein = load_data18()

    # Check if data was loaded properly
    if any([disease is None, drug is None, metabolite is None, mirna is None, mrna is None, protein is None]):
        st.stop()  # Stop if data loading failed

    # Merge all data
    merged_data = merge_data(disease, drug, metabolite, mirna, mrna, protein)

    # User Input
    st.write("Enter a query to explore relationships across drugs, diseases, metabolites, miRNAs, mRNAs, and proteins.")
    st.markdown("---")

    input_type = st.selectbox("Select input type:", ['Drug Name', 'Disease', 'Metabolite', 'miRNA', 'mRNA', 'Gene Name', 'Protein Name (UniProt ID)'])
    user_input = st.text_input(f"Enter {input_type}:")

    # Submit button to process analysis
    if st.button("Submit"):
        # Check if the input type is a valid column in merged_data
        if input_type in merged_data.columns:
            # Filter based on user input, handling NaN values
            filtered_data = merged_data[merged_data[input_type].fillna('').str.contains(user_input, na=False, case=False)]
            
            if filtered_data.empty:
                st.write("No data found. Please try a different input.")
            else:
                st.write(f"Results for {input_type} '{user_input}':")
                st.dataframe(filtered_data)

                # Build the knowledge graph
                G, node_colors = build_knowledge_graph18(filtered_data, node_categories)  # Pass node_categories here
                
                # Plot the interactive knowledge graph (pass both G and node_colors)
                st.plotly_chart(plot_interactive_knowledge_graph18(G, node_colors, node_categories))

                # Perform additional analysis
                perform_analysis(filtered_data)
        else:
            st.write("Invalid input type. Please try again.")
    
    st.markdown("---")
    st.write("""**About DrugSwitch**""")
    st.write("""
        **DrugSwitch** integrates multi-omics data including transcriptomes, proteomes, and metabolomes
        to predict drug-disease relationships using deep learning algorithms. This app provides users the
        ability to explore drug repurposing opportunities.
    """)

# Tool 19: DrugEnricher - Drug Target and Pathway Enrichment Analysis
elif selected_tool == "Tool 19: DrugEnricher":
    st.markdown("<h4 style='color: #3498db;'> Tool 19: DrugEnricher - Drug Pathway Enrichment Analysis</h4>", unsafe_allow_html=True)
    
    st.markdown("---")
    # Load the datasets
    combine_score, path2gene_wikipathway = load_data19()
    
    # Normalize the column names by stripping whitespace
    combine_score.columns = combine_score.columns.str.strip()
    path2gene_wikipathway.columns = path2gene_wikipathway.columns.str.strip()
    
    # Sidebar for user input and submit button
    st.write("Input Drug for Enrichment Analysis")
    drug_input = st.text_input("Enter a Drug Name:", "Prednisolone")
    
    # Checkbox for top N enrichment options
    st.write("Select Top N Pathway Enrichment Options for Plot:")
    top_10 = st.checkbox("Top 10", value=True)
    top_20 = st.checkbox("Top 20")
    top_30 = st.checkbox("Top 30")
    top_40 = st.checkbox("Top 40")
    top_50 = st.checkbox("Top 50")
    
    # Submit button to trigger analysis
    if st.button("Submit"):
    
        # Filter the combine_score dataset for the selected drug
        filtered_drug_data = combine_score[combine_score['Drug'].str.lower() == drug_input.lower()]
    
        if filtered_drug_data.empty:
            st.warning(f"No data found for drug: {drug_input}. Please enter a valid drug name.")
        else:
            st.success(f"Results found for drug: {drug_input}")
    
            # Show the table for Drug and associated Targets
            st.header(f"Targets associated with {drug_input}")
            st.write("Here is the list of targets associated with the selected drug, along with the enrichment scores:")
            
            # Rename the 'Combine_score' to 'DrugEnricher Score'
            filtered_drug_data = filtered_drug_data.rename(columns={'Combine_score': 'DrugEnricher Score'})
            
            # Display the table
            st.dataframe(filtered_drug_data[['Drug', 'Target', 'DrugEnricher Score']])
    
            # Set the default to show the top 50 targets
            top_n_targets = 50
            
            # Sort the data by score and select top 50 targets
            filtered_drug_data_top_50 = filtered_drug_data.sort_values(by='DrugEnricher Score', ascending=False).head(top_n_targets)
            
            # Create an interactive horizontal bar plot using Plotly for the top 50 targets
            fig = px.bar(
                filtered_drug_data_top_50,
                x='DrugEnricher Score',
                y='Target',
                orientation='h',
                color='DrugEnricher Score',
                color_continuous_scale='Blues',
                title=f"DrugEnricher Score Distribution for Top {top_n_targets} Targets of {drug_input}",
                labels={'DrugEnricher Score': 'DrugEnricher Score', 'Target': 'Target Name'}
            )
            
            # Customize layout for better readability and spacing
            fig.update_layout(
                xaxis_title="DrugEnricher Score",
                yaxis_title="Target",
                title_x=0.5,  # Center the title
                height=800,  # Adjust height for better spacing with 50 targets
                margin=dict(l=150, r=20, t=80, b=50),  # Adjust margins to avoid cutting labels
                yaxis=dict(tickmode='linear'),  # Ensure all target names are displayed
                yaxis_tickangle=0,  # Keep labels horizontal to avoid overlap
            )
            
            # Display the interactive plot in Streamlit
            st.plotly_chart(fig)
    
    
            # Perform pathway enrichment analysis by matching Targets with Pathways
            st.header("Pathway Enrichment Analysis")
            st.write(f"Pathway enrichment analysis based on the target genes of the drug {drug_input}.")
            
            # Split the gene lists in Path2gene_wikipathway and explode them into separate rows
            path2gene_wikipathway['Gene_List'] = path2gene_wikipathway['Genes'].str.split(',')
            path2gene_wikipathway = path2gene_wikipathway.explode('Gene_List')
            path2gene_wikipathway['Gene_List'] = path2gene_wikipathway['Gene_List'].str.strip()
            
            # Get the unique target genes for the drug
            drug_targets = filtered_drug_data['Target'].unique()
            
            # Find matching pathways for the drug targets
            pathway_matches = path2gene_wikipathway[path2gene_wikipathway['Gene_List'].isin(drug_targets)]
            
            if pathway_matches.empty:
                st.warning(f"No matching pathways found for the targets of {drug_input}.")
            else:
                st.subheader(f"Matching Pathways for the Targets of {drug_input}")
                st.write("The following table shows the pathways associated with the drug targets:")
                
                st.dataframe(pathway_matches[['Path', 'Description', 'Gene_List']].drop_duplicates())
                
                # Determine which enrichment top N to display
                top_n = 0
                if top_10:
                    top_n = 10
                elif top_20:
                    top_n = 20
                elif top_30:
                    top_n = 30
                elif top_40:
                    top_n = 40
                elif top_50:
                    top_n = 50
    
                # Pathway enrichment plot (Dot Plot)
                st.subheader(f"Top {top_n} Pathway Enrichment (Dot Plot)")
                pathway_counts = pathway_matches['Path'].value_counts().head(top_n).reset_index()
                pathway_counts.columns = ['Pathway', 'Count']
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=pathway_counts, x='Count', y='Pathway', s=100, color='orange', ax=ax)
                for i in range(pathway_counts.shape[0]):
                    ax.text(pathway_counts['Count'].iloc[i], pathway_counts['Pathway'].iloc[i], pathway_counts['Count'].iloc[i], 
                            horizontalalignment='left', size='medium', color='black', weight='semibold')
                ax.set_xlabel("Number of Associated Targets", fontsize=14)
                ax.set_ylabel("Pathway", fontsize=14)
                ax.set_title(f"Top {top_n} Pathway Enrichment for {drug_input}", fontsize=16)
                st.pyplot(fig)
    
                # Pathway enrichment plot (Bar Plot)
                st.subheader(f"Top {top_n} Pathway Enrichment (Bar Plot)")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Count', y='Pathway', data=pathway_counts, palette='coolwarm', ax=ax)
                ax.set_xlabel("Number of Associated Targets", fontsize=14)
                ax.set_ylabel("Pathway", fontsize=14)
                ax.set_title(f"Top {top_n} Pathway Enrichment for {drug_input} (Bar Plot)", fontsize=16)
                st.pyplot(fig)



    st.markdown("---")
    st.write("""**About DrugEnricher**""")
    st.write("""
        **DrugEnricher** is a tool for performing pathway enrichment analysis based on Drug. It helps to identify potential pathways related to a drug by analyzing associated targets and their enrichment across biological pathways.
    """)


# Tool 20: DrugMarker - Tool designed to assist researchers in screening and analyzing pharmacogenomic Biomarkers associated with Drugs
elif selected_tool == "Tool 20: DrugMarker":
    st.markdown("<h4 style='color: #3498db;'> Tool 20: DrugMarker -Screening Biomarkers associated Drugs</h4>", unsafe_allow_html=True)
    
    st.markdown("---")
    # Load the datasets
    biomarker_data = load_data20()
    
    # Sidebar for user input
    st.header("Input Drug for Biomarker Screening")
    
    # Use selectbox for drug selection
    #drug_list = biomarker_data['Drug'].unique()  # Get unique drugs from the dataset
    #drug_input = st.selectbox("Select a Drug Name:", drug_list)
    # Use selectbox for drug selection with default value 'Olaparib'
    drug_list = biomarker_data['Drug'].unique()  # Get unique drugs from the dataset
    drug_input = st.selectbox("Select a Drug Name:", drug_list, index=list(drug_list).index("Olaparib"))
    
    # Submit button to trigger analysis
    if st.button("Submit"):
        # Check if the biomarker_data is empty
        if biomarker_data.empty:
            st.warning("No data loaded. Please check the source.")
        else:
            # Filter the dataset for the selected drug
            filtered_data = biomarker_data[biomarker_data['Drug'].str.lower() == drug_input.lower()]
    
            if filtered_data.empty:
                st.warning(f"No data found for drug: {drug_input}. Please try another drug.")
            else:
                st.success(f"Results found for drug: {drug_input}")
    
                # Display the filtered data
                #st.header(f"Biomarkers associated with {drug_input}")
                st.dataframe(filtered_data[['Drug', 'TherapeuticArea', 'Biomarker', 'LabelingSections']])
    
                # Biomarker Distribution Plot
                #st.subheader(f"Biomarkers Distribution for {drug_input}")
                biomarker_counts = filtered_data['Biomarker'].value_counts().reset_index()
                biomarker_counts.columns = ['Biomarker', 'Count']
    
                fig = px.bar(biomarker_counts, x='Biomarker', y='Count',
                             color='Count', title=f"Biomarker Distribution for {drug_input}",
                             color_continuous_scale='Viridis', text='Count')
    
                fig.update_traces(texttemplate='%{text}', textposition='outside')
                fig.update_layout(yaxis_title='Count', xaxis_title='Biomarker', height=400)
                st.plotly_chart(fig)
    
                # Labeling Sections Distribution Plot
                #st.subheader("Therapeutic Biomarkers Description Distribution")
                labeling_counts = filtered_data['LabelingSections'].str.split(', ').explode().value_counts().reset_index()
                labeling_counts.columns = ['Description', 'Count']
    
                fig2 = px.bar(labeling_counts, x='Description', y='Count',
                              color='Count', title="Therapeutic Biomarkers Description Distribution",
                              color_continuous_scale='Plasma', text='Count')
    
                fig2.update_traces(texttemplate='%{text}', textposition='outside')
                fig2.update_layout(yaxis_title='Count', xaxis_title='Therapeutic Biomarkers Description', height=400)
                st.plotly_chart(fig2)
    
                # Therapeutic Area Distribution Plot
                st.subheader("Therapeutic Area Distribution")
                therapeutic_counts = filtered_data['TherapeuticArea'].value_counts().reset_index()
                therapeutic_counts.columns = ['Therapeutic Area', 'Count']
    
                fig3 = px.pie(therapeutic_counts, names='Therapeutic Area', values='Count',
                              title="Therapeutic Area Distribution", color_discrete_sequence=px.colors.sequential.RdBu)
    
                st.plotly_chart(fig3)
                st.write("""**About DrugMarker**""")
                st.write("""
                **DrugMarker** is a powerful tool designed to assist researchers in screening and analyzing pharmacogenomic biomarkers associated with drugs. 
                This tool allows users to easily access biomarkers, explore their therapeutic areas, and visualize data with stunning, publication-ready plots.
                """)
    
    # Footer with tool description
    #st.title("About DrugMarker")
    st.write("""
    This tool aims to simplify the screening of biomarkers associated with various drugs, helping researchers visualize and analyze crucial pharmacogenomic data effectively.
    """)


# Team Section
elif selected_tool == "Team":
    st.markdown("""
    <div style='margin: 30px 0; padding: 20px; background: #f9f9f9; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);'>
        <h4 style='color: #3498db; text-align: center; margin-bottom: 20px;'>CSIR-CIMAP, Lucknow Team</h4>
    </div>
    """, unsafe_allow_html=True)

    developer_data = [
        {"name": "Prabodh Kumar Trivedi", "Position": "Director", "email": "director@cimap.res.in", "photo_url": "https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/Images/cimap_director.jpg"},
        {"name": "Dr. Sumya Pathak", "Position": "Senior Scientist", "email": "sumyapathak@cimap.res.in", "photo_url": "https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/Images/769.jpg"},
        {"name": "Dr. Aman Kaushik", "Position": "Scientist", "email": "amankaushik@cimap.res.in", "photo_url": "https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/Images/ACK-2.jpg"}
    ]

    for dev in developer_data:
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin-bottom: 20px;">
            <img src="{dev['photo_url']}" alt="{dev['name']}" style="width: 100px; height: 100px; border-radius: 50%; margin-right: 20px;">
            <div>
                <h5 style="color: #2c3e50; margin-bottom: 5px;">{dev['name']}</h5>
                <p style="color: #3498db; font-weight: bold; margin: 0;">{dev['Position']}</p>
                <p style="color: #7f8c8d; margin-top: 5px;">Email: <a href="mailto:{dev['email']}">{dev['email']}</a></p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# About Section
elif selected_tool == "About":
    st.markdown("""
    <div style='margin: 30px 0; padding: 20px; background: #f9f9f9; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);'>
        <h4 style='color: #3498db; text-align: center; margin-bottom: 20px;'>About PhytoChemX Modules</h4>
    </div>
    """, unsafe_allow_html=True)
    # Add the 6 image sections with clickable links and headings
    col1, col2 = st.columns(2)

    # Create columns for layout with equal width (1:1 ratio)
    col1, col2 = st.columns([1, 1])  # Equal width for text and images
    
    with col1:
        st.markdown("**PhytoChemX**")
        st.markdown(
            """
            <div style="text-align: center; border: 1px solid #ddd; padding: 15px; border-radius: 10px; margin-bottom: 20px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); transition: transform 0.3s;">
                <img src="https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/Modules.png" width="300">
                <p style="text-align: justify; margin-top: 10px;">An advanced platform to explore phytochemicals, analyze their properties, and study medicinal potential.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col2:
        st.markdown("**Modules**")
        st.markdown(
            """
            <div style="text-align: center; border: 1px solid #ddd; padding: 15px; border-radius: 10px; margin-bottom: 20px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); transition: transform 0.3s;">
                <img src="https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/Modules2.png" width="300">
                <p style="text-align: justify; margin-top: 10px;">Designed for all levels, our user-friendly interface and tutorials streamline research processes.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("""
    <div style='font-size: 14px; text-align: justify; margin: 0 auto; max-width: 800px;'>
        <p>
            <strong>Tool 1: DataExtract</strong> - Extract and profile molecular fingerprints for in-depth analysis.<br>
            <strong>Tool 2: Ro5Scan</strong> - Evaluate compounds against Lipinski's Rule of Five for drug-likeness.<br>
            <strong>Tool 3: Descripto</strong> - Calculate and visualize molecular descriptors for better insights.<br>
            <strong>Tool 4: SimAnalyzr</strong> - Compare molecular fingerprints to assess similarity and relationships.<br>
            <strong>Tool 5: ChemCluster</strong> - Cluster compounds based on their molecular features.<br>
            <strong>Tool 6: MolInsight</strong> - Gain insights into individual molecules with advanced substructure analysis.<br>
            <strong>Tool 7: ScreenGen-ML</strong> - Perform machine learning-based screening for compound evaluation.<br>
            <strong>Tool 8: ScreenGen-DL</strong> - Utilize deep learning for enhanced compound screening and prediction.<br>
            <strong>Tool 9: TargetPredictor</strong> - Predict interactions between pharmacodynamic components and targets.<br>
            <strong>Tool 10: SmartTox</strong> - Assess drug interactions and potential toxicities using machine learning models.<br>
            <strong>Tool 11: DDI-Predictor</strong> - Predict drug-drug interactions based on molecular structures.<br>
            <strong>Tool 12: CellDrugFinder</strong> - Identify essential drugs for specific tissues and cell lines.<br>
            <strong>Tool 13: NetPharm</strong> - Explore network pharmacology for target identification and drug repositioning.<br>
            <strong>Tool 14: QSARGx</strong> - Build and evaluate QSAR models to predict biological activities based on 2D structure.<br>
            <strong>Tool 15: ChemSyn</strong> - This tool enables users to apply various chemical transformations to SMILES strings.<br>
            <strong>Tool 16: ChemPROTAC</strong> - Deep Learning Driven Server for PROTAC Prediction.<br>
            <strong>Tool 17: NanoMedScore</strong> - Allows users to predict the pharmacokinetics of nanoparticles using ML models.<br>
            <strong>Tool 18: DrugSwitch</strong> - This module provides users the ability to explore Drug Repurposing opportunities.<br>
            <strong>Tool 19: DrugEnricher</strong> - Tool for performing Pathway Enrichment analysis based on Drugs.<br>
            <strong>Tool 20: DrugMarker</strong> - Screening and analyzing pharmacogenomic biomarkers associated with Drugs.<br>
            Explore these tools to enhance your research and gain valuable molecular insights.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    
# Advanced Streamlit Styling with color "#87CEFA" and light yellow border for figures
st.markdown("""
<style>
    /* Main background color */
    .css-1d391kg {background-color: #87CEFA;}
    
    /* Header font styling */
    .css-1vgnld4 {
        font-family: 'Trebuchet MS', sans-serif;
        color: #ffffff;
        font-weight: bold;
    }

    /* Button styling */
    .stButton>button {
        background-color: #4A90E2;
        color: #ffffff;
        border-radius: 12px;
        padding: 10px 24px;
        font-size: 16px;
        font-weight: bold;
        border: none;
        transition: background-color 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #007ACC;
    }

    /* Adding box shadows to containers */
    .css-1lcbmhc {
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        border-radius: 10px;
    }

    /* Sidebar styling (if applicable) */
    .css-1d3fdmu {
        background-color: #007ACC;
        color: #ffffff;
        font-family: 'Verdana';
    }

    /* Enhancing text input fields */
    .stTextInput, .stTextArea {
        background-color: #F0F8FF;
        border-radius: 8px;
        border: 2px solid #4A90E2;
        color: #333;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Figure border styling */
    .stPlotlyChart, .stImage, .stAltairChart, .stVegaLiteChart, .stDataFrame {
        border: 3px solid #FFFFE0; /* Light yellow border */
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        background-color: #ffffff;
    }

    /* Footer styling */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #007ACC;
        color: white;
        text-align: center;
        padding: 10px;
        font-family: 'Arial', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# Streamlit App Code (No need for main())
#st.title("My Streamlit App")

# Display home page content
#st.write("Welcome to the home page of the app!")

# Call the background fetch function
fetch_monitor_in_background()


# Footer Section
st.markdown(
    """
    <div class='footer' style="background-color: #0000FF; padding: 10px; border-radius: 10px; text-align: center;">
        <p style="font-size: 10px; color: white; text-align: center; margin: 5px 0; line-height: 1.4;">
            <strong>Disclaimer:</strong> Predictions on this web server are generated using computational models. While we aim for accuracy, results may not always be precise and can differ from experiments. 
        </p>
        <p style="font-size: 10px; color: white; text-align: center; margin: 5px 0; line-height: 1.4;">
            Users should verify findings experimentally before drawing conclusions. We are not responsible for inaccuracies or any consequences of using these predictions.  
        </p>
        <p style="font-size: 10px; color: white; margin-top: 5px;">
            PhytoChemX is for research purposes only. Developed by Dr. Aman Kaushik (Scientist, CSIR-CIMAP, Lucknow).
        </p>
        <p style="font-size: 10px; color: white; margin-top: 5px; font-weight: bold;">
            © 2024 CSIR-CIMAP. All rights reserved.  
        </p>
    </div>
    """, 
    unsafe_allow_html=True
)


# Logo Carousel
st.markdown(
    """
    <style>
        .logo-carousel {
            width: 100%;
            background-color: #007bff; /* Blue stripe */
            padding: 10px 0;
            overflow: hidden;
            white-space: nowrap;
            position: relative;
        }
        .logo-carousel .logos {
            display: flex;
            animation: scroll 15s linear infinite;
            width: max-content; /* Ensures continuous looping */
        }
        .logo-carousel .logos img {
            height: 50px;
            width: auto;
            margin: 0 15px;
            transition: transform 0.3s;
        }
        .logo-carousel .logos img:hover {
            transform: scale(1.1);
        }
        @keyframes scroll {
            from { transform: translateX(0); }
            to { transform: translateX(-50%); }
        }
    </style>

    <div class="logo-carousel">
        <div class="logos">
            <img src="https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/CSIR-Logo.jpg" alt="CSIR Logo">
            <img src="https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/CIMAP.jpeg" alt="CIMAP">
            <img src="https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/DigitalIndia.png" alt="Digital India">
            <img src="https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/MakeinIndia.png" alt="Make in India">
            <img src="https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/SwachhBharat.png" alt="Swachh Bharat">
            <img src="https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/SkillIndia.png" alt="Skill India">
            <img src="https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/NationalHealthMission.png" alt="National Health Mission">
            <img src="https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/G20-Presidency.jpeg" alt="G20 Presidency">
            <img src="https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/OneDigitalIndia.png" alt="One Digital India">
            <img src="https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/MadeInIndia.jpeg" alt="Made In India">
            <img src="https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/AYUSH.jpeg" alt="AYUSH">
            <img src="https://raw.githubusercontent.com/amanbioinfo/ML_based_Scaffold_Prediction/main/mahakumbh.jpeg" alt="Mahakumbh">
        </div>
    </div>
    """, 
    unsafe_allow_html=True
)
