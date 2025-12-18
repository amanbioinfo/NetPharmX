import streamlit as st
import pandas as pd
from pathlib import Path
import logging
from scripts import (
    get_smiles,
    predict_targets,
    targets_to_disease,
    enrichment_analysis,
    network_plot,
    compound_target_heatmap,
    disease_class_bubble,
    target_pathway_heatmap,
    sankey_plot,
    target_centrality_radar,
    chem_similarity_vs_target_overlap,
    mechanism_cartoon,
    docking_score_plot,
)
from scripts_ppi import string_ppi_expand, hub_genes, ppi_network_plot, hub_gene_barplot

# ========================
# Logging setup
# ========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
st.set_page_config(page_title="Compound→Target→Disease Pipeline", layout="wide")

st.title("🧬 Compound → Target → Disease Pipeline")

# ========================
# Directories
# ========================
BASE_DIR = Path.cwd() / "user_pipeline_results"
INPUT_DIR = BASE_DIR / "input"
TARGETS_DIR = BASE_DIR / "targets"
DISEASES_DIR = BASE_DIR / "diseases"
PLOTS_DIR = BASE_DIR / "plots"
ENRICH_DIR = BASE_DIR / "enrichment"
PPI_DIR = BASE_DIR / "ppi"

for d in [INPUT_DIR, TARGETS_DIR, DISEASES_DIR, PLOTS_DIR, ENRICH_DIR, PPI_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ========================
# Upload compounds
# ========================
st.sidebar.header("Input Compounds")
input_method = st.sidebar.radio("Input method", ["Manual entry", "CSV upload"])

if input_method == "Manual entry":
    compounds_text = st.sidebar.text_area("Enter compounds (comma separated)", "Ursolic acid, Oleanolic acid")
    compounds_list = [c.strip() for c in compounds_text.split(",")]
    input_df = pd.DataFrame({"Compound": compounds_list})
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV with 'Compound' column", type="csv")
    if uploaded_file:
        input_df = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload a CSV file to proceed.")
        st.stop()

input_csv = INPUT_DIR / "compounds.csv"
input_df.to_csv(input_csv, index=False)
st.success(f"✔ Saved compounds to {input_csv}")

# ========================
# Pipeline steps
# ========================
try:
    st.info("Step 1: Fetching compound SMILES")
    compound_smiles_csv = TARGETS_DIR / "compound_smiles.csv"
    get_smiles.main(str(input_csv), str(compound_smiles_csv))
    st.success("✔ SMILES fetched")

    st.info("Step 2: Predicting targets")
    compound_targets_csv = TARGETS_DIR / "compound_targets.csv"
    predict_targets.main(str(compound_smiles_csv), str(compound_targets_csv))
    st.success("✔ Targets predicted")

    st.info("Step 3: Mapping targets to diseases")
    compound_diseases_csv = DISEASES_DIR / "compound_diseases.csv"
    targets_to_disease.main(str(compound_targets_csv), str(compound_diseases_csv))
    st.success("✔ Targets mapped to diseases")

    st.info("Step 4: Enrichment analysis")
    enrichment_analysis.main(str(compound_targets_csv), str(ENRICH_DIR))
    st.success("✔ Enrichment analysis complete")

    st.info("Step 5: Compound-Target-Disease network")
    network_plot.main(str(compound_targets_csv), str(compound_diseases_csv), str(PLOTS_DIR))
    st.success("✔ Network plot generated")

    st.info("Step 6: STRING PPI expansion")
    ppi_edges_csv = PPI_DIR / "ppi_edges.csv"
    string_ppi_expand.main(str(compound_targets_csv), str(ppi_edges_csv))
    st.success("✔ PPI edges expanded")

    st.info("Step 7: Hub gene analysis")
    hub_genes_csv = PPI_DIR / "hub_genes.csv"
    hub_genes.main(str(ppi_edges_csv), str(hub_genes_csv))
    st.success("✔ Hub genes identified")

    st.info("Step 8: PPI network plotting")
    ppi_network_plot.main(str(ppi_edges_csv), str(PLOTS_DIR))
    st.success("✔ PPI network plotted")

    st.info("Step 9: Compound-Target heatmap")
    compound_target_heatmap.main(str(compound_targets_csv), str(PLOTS_DIR))
    st.success("✔ Compound-target heatmap plotted")

    st.info("Step 10: Hub gene barplot")
    hub_gene_barplot.main(str(hub_genes_csv), str(PLOTS_DIR))
    st.success("✔ Hub gene barplot generated")

    st.info("Step 11: Disease class bubble plot")
    disease_class_bubble.main(str(compound_diseases_csv), str(PLOTS_DIR))
    st.success("✔ Disease class bubble plot generated")

    st.info("Step 12: Target-Pathway heatmap")
    target_pathway_heatmap.main(str(compound_targets_csv), str(PLOTS_DIR))
    st.success("✔ Target-pathway heatmap generated")

    st.info("Step 13: Sankey plot")
    sankey_plot.main()
    st.success("✔ Sankey plot generated")

    st.info("Step 14: Target centrality radar / degree")
    target_centrality_radar.main()
    st.success("✔ Centrality radar / degree plot generated")

    st.info("Step 15: Chemical similarity vs target overlap")
    chem_similarity_vs_target_overlap.main()
    st.success("✔ Chemical similarity plot generated")

    st.info("Step 16: Mechanism-of-action cartoon")
    mechanism_cartoon.main()
    st.success("✔ Mechanism cartoon generated")

    st.info("Step 17: Docking score distribution (optional)")
    docking_score_plot.main()
    st.success("✔ Docking score distribution plotted")

except Exception as e:
    st.error(f"❌ Error occurred: {e}")

st.success("🎉 Pipeline completed successfully! All plots are saved in the 'plots' folder.")
