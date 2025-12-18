import streamlit as st
from pathlib import Path
import pandas as pd

# Import all pipeline functions
from scripts.get_smiles import get_smiles
from scripts.predict_targets import predict_targets
from scripts.targets_to_disease import map_targets_to_disease
from scripts.enrichment_analysis import plot_enrichment
from scripts.network_plot import plot_network
from scripts.scripts_ppi.string_ppi_expand import string_ppi_expand
from scripts.scripts_ppi.hub_genes import plot_hub_genes
from scripts.scripts_ppi.ppi_network_plot import plot_ppi
from scripts.compound_target_heatmap import plot_heatmap
from scripts.disease_class_bubble import plot_disease_bubble
from scripts.target_pathway_heatmap import plot_target_pathway_heatmap
from scripts.sankey_plot import plot_sankey
from scripts.target_centrality_radar import plot_target_centrality
from scripts.chem_similarity_vs_target_overlap import plot_chem_similarity
from scripts.mechanism_cartoon import plot_mechanism
from scripts.docking_score_plot import plot_docking_scores

# Output folders
BASE_DIR = Path("Streamlit_Output")
BASE_DIR.mkdir(exist_ok=True)
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Network Pharmacology Pipeline", layout="wide")
st.title("🌿 Network Pharmacology Pipeline for User-Entered Compounds")
st.write("Enter compounds and get dynamic, publication-ready plots for target, disease, enrichment, and PPI analysis.")

# 1️⃣ User Input
compounds_input = st.text_area(
    "Enter compounds (one per line):",
    value="Ursolic acid\nOleanolic acid\nBetulinic acid"
)

if st.button("Run Pipeline"):
    if not compounds_input.strip():
        st.error("Please enter at least one compound!")
    else:
        # Save input compounds to CSV
        compounds = [c.strip() for c in compounds_input.split("\n") if c.strip()]
        input_csv = BASE_DIR / "input_compounds.csv"
        pd.DataFrame({'Compound': compounds}).to_csv(input_csv, index=False)

        st.info("Running pipeline...")
        
        # 2️⃣ Step 1: Get SMILES
        smiles_csv = BASE_DIR / "compound_smiles.csv"
        df_smiles = get_smiles(input_csv, smiles_csv)
        st.success("✅ SMILES fetched")
        st.dataframe(df_smiles)

        # 3️⃣ Step 2: Predict Targets
        targets_csv = BASE_DIR / "compound_targets.csv"
        df_targets = predict_targets(smiles_csv, targets_csv)
        st.success("✅ Targets predicted")
        st.dataframe(df_targets)

        # 4️⃣ Step 3: Map Targets to Disease
        disease_csv = BASE_DIR / "compound_diseases.csv"
        df_disease = map_targets_to_disease(targets_csv, disease_csv)
        st.success("✅ Targets mapped to diseases")
        st.dataframe(df_disease)

        # 5️⃣ Step 4: Enrichment Analysis
        fig_enrich = plot_enrichment(targets_csv, PLOTS_DIR)
        st.success("✅ Enrichment analysis complete")
        st.pyplot(fig_enrich)

        # 6️⃣ Step 5: Compound–Target–Disease Network
        fig_network = plot_network(targets_csv, disease_csv, PLOTS_DIR)
        st.success("✅ Compound–Target–Disease network plotted")
        st.pyplot(fig_network)

        # 7️⃣ Step 6: STRING PPI Expansion
        ppi_edges_csv = BASE_DIR / "ppi_edges.csv"
        df_ppi = string_ppi_expand(targets_csv, ppi_edges_csv)
        st.success("✅ PPI expansion done")
        st.dataframe(df_ppi)

        # 8️⃣ Step 7: Hub Gene Analysis
        fig_hub = plot_hub_genes(ppi_edges_csv, PLOTS_DIR)
        st.success("✅ Hub gene barplot generated")
        st.pyplot(fig_hub)

        # 9️⃣ Step 8: PPI Network Plot
        fig_ppi = plot_ppi(ppi_edges_csv, PLOTS_DIR)
        st.success("✅ PPI network plotted")
        st.pyplot(fig_ppi)

        # 🔟 Step 9: Compound–Target Heatmap
        fig_heatmap = plot_heatmap(targets_csv, PLOTS_DIR)
        st.success("✅ Compound–Target heatmap generated")
        st.pyplot(fig_heatmap)

        # 1️⃣1️⃣ Disease Class Bubble Plot
        fig_bubble = plot_disease_bubble(disease_csv, PLOTS_DIR)
        st.success("✅ Disease class bubble plot generated")
        st.pyplot(fig_bubble)

        # 1️⃣2️⃣ Target–Pathway Heatmap
        # Using targets CSV as dummy enrichment input for now
        fig_pathway = plot_target_pathway_heatmap(targets_csv, PLOTS_DIR)
        st.success("✅ Target–Pathway heatmap generated")
        st.pyplot(fig_pathway)

        # 1️⃣3️⃣ Sankey Multi-layer Network
        fig_sankey = plot_sankey(targets_csv, disease_csv, PLOTS_DIR)
        st.success("✅ Sankey diagram generated")
        st.write(fig_sankey)

        # 1️⃣4️⃣ Target Centrality Radar / Degree Barplot
        fig_centrality = plot_target_centrality(ppi_edges_csv, PLOTS_DIR)
        st.success("✅ Target centrality plot generated")
        st.pyplot(fig_centrality)

        # 1️⃣5️⃣ Chemical Similarity vs Target Overlap
        fig_chem = plot_chem_similarity(targets_csv, PLOTS_DIR)
        st.success("✅ Chemical similarity vs target overlap plot generated")
        st.pyplot(fig_chem)

        # 1️⃣6️⃣ Mechanism-of-Action Cartoon
        fig_mech = plot_mechanism(PLOTS_DIR)
        st.success("✅ Mechanism cartoon generated")
        st.pyplot(fig_mech)

        # 1️⃣7️⃣ Docking Score Distribution (optional)
        fig_dock = plot_docking_scores(None, PLOTS_DIR)
        st.success("✅ Docking score distribution plotted (dummy)")
        st.pyplot(fig_dock)

        # 1️⃣8️⃣ Any extra plots/steps can be added similarly

        st.success("🎉 Pipeline completed! All plots saved in `Streamlit_Output/plots` as PNG + SVG.")
