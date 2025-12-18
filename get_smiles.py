import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

def get_smiles(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df['SMILES'] = df['Compound'].apply(lambda x: Chem.MolToSmiles(AllChem.MolFromName(x)) if AllChem.MolFromName(x) else None)
    df.to_csv(output_csv, index=False)
    return df
