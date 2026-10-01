# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 15:00:19 2025

@author: medisp-2
"""

from rdkit import Chem
from rdkit.Chem import MolToSmiles
import os
import pandas as pd
from typing import List, Dict, Union

# --- Function definition remains the same as before ---

def extract_data_from_sdf(
    sdf_filepath: str, 
    prop_names: List[str], # Now explicitly using the user-provided list
    smiles_col_name: str = 'SMILES'
) -> Union[pd.DataFrame, None]:
    """
    Reads an SDF file, extracts the canonical SMILES string and specified 
    properties (like MolPort ID and RMSD) for each molecule.
    """
    
    if not os.path.exists(sdf_filepath):
        print(f"Error: SDF file not found at {sdf_filepath}")
        return None
    
    data_records: List[Dict] = []
    
    # 2. Use SDMolSupplier to read molecules
    suppl = Chem.SDMolSupplier(sdf_filepath, sanitize=True, removeHs=True)
    
    print(f"Processing molecules from: {sdf_filepath}")
    
    # 3. Iterate and extract
    for i, mol in enumerate(suppl):
        record = {}
        
        if mol is None:
            print(f"Warning: Skipping invalid or empty molecule at record index {i}")
            continue
            
        try:
            # A. Extract SMILES
            record[smiles_col_name] = MolToSmiles(mol, canonical=True, isomericSmiles=True)
            
            # B. Extract Specified Properties
            for prop in prop_names:
                if mol.HasProp(prop):
                    # mol.GetProp(prop) extracts the value of the property tag
                    record[prop] = mol.GetProp(prop)
                else:
                    # Use a placeholder if the property is missing for a molecule
                    record[prop] = None 
                    
            data_records.append(record)
            
        except Exception as e:
            print(f"Error processing molecule {i}: {e}")
            
    print(f"Successfully extracted data for {len(data_records)} valid molecules.")
    
    # 4. Convert the list of dictionaries into a clean Pandas DataFrame
    return pd.DataFrame(data_records)

#===============================================================================
#==================================MAIN PROGRAM ================================
#===============================================================================
import moduleUtils as U
U.cls()
my_file='test_file_hits.sdf'
path=''    
SDF_FILE=path+my_file
# SDF_FILE = 'query_results.sdf'
PROPERTY_KEYS = ['_Name', 'rmsd'] #'MolPort',
# 1. Call the function
df_results = extract_data_from_sdf(SDF_FILE, prop_names=PROPERTY_KEYS)

# 2. Print the results DataFrame
if df_results is not None:
    print("\n--- Extracted Data DataFrame (MolPort and rmsd columns) ---")
    print(df_results.head())
    df_results.to_csv(SDF_FILE[:-3]+'csv', index=False)
    df_results.to_excel(SDF_FILE[:-3]+'xlsx', index=False)
    