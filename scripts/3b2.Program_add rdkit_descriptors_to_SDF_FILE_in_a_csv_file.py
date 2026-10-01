# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:55:12 2025

@author: medisp-2
"""

#PROGRAM TO CALCULATE THE DESCRIPTORS FROM THE MOLECULE SMILES USING THE RDKIT TOOL
import moduleUtils as U
import pandas as pd
import os
try:
    import rdkit
    print("module 'rdkit' is installed")
except ModuleNotFoundError:
    print("module 'rdkit' is not installed")
    # or
    os.system("pip install rdkit")


from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
import os
U.cls()
'''# Function to calculate descriptors for each SMILES csv column'''
def calculate_descriptors(smiles,calculator):
    if isinstance(smiles, str) and smiles.strip(): # Check if SMILES is a non-empty string
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return calculator.CalcDescriptors(mol)
    return [None] * len(descriptor_names) # Return None for invalid SMILES

#-------------------------------------------------------------------
# function to anounce the end of the ML design
def end_sound(dur):
    import winsound
    duration = dur  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)


#-----------------------------------------------------------
#-------------------- MAIN PROGRAM -------------------------
#-----------------------------------------------------------


U.cls()
t1=U.tic()
'''1.Read the data file with no descriptors'''
path1=''
my_file='test_file_hits.csv'
File=path1+my_file

  
print('\n processing file: %s ' % File)
print(' please wait.....')
data = pd.read_csv(File)

# U.RETURN()
# Get all available descriptors
descriptor_names = [desc_name[0] for desc_name in Descriptors._descList]
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
'''# 1. Extract SMILES column'''
smiles_list = data['SMILES']

'''#2. Apply the descriptor calculation'''
descriptors = [calculate_descriptors(smiles,calculator) for smiles in smiles_list]

'''#3. Create a new DataFrame with descriptors'''
descriptor_df = pd.DataFrame(descriptors, columns=descriptor_names)

'''#4. Concatenate original data with descriptor data'''
final_data = pd.concat([data, descriptor_df], axis=1)

'''#5. Save the result to a new csv file'''
df=final_data
df=df.dropna() 
df=df.fillna(0)
df.to_csv(File[:-4]+'_descriptors.csv',index=False)
df.to_excel(File[:-4]+'_descriptors.xlsx',index=False)
print('\n file %s has been created: ' % (File[:-4]+'_descriptors.csv') )
print('\n**********DONE************')
U.toc(t1,'time taken')
end_sound(1000)