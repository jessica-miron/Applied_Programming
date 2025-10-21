##########################################################
## BME 494/598:  Homework 3.1 Python code               ##
##  ______     ______     __  __                        ##
## /\  __ \   /\  ___\   /\ \/\ \                       ##
## \ \  __ \  \ \___  \  \ \ \_\ \                      ##
##  \ \_\ \_\  \/\_____\  \ \_____\                     ##
##   \/_/\/_/   \/_____/   \/_____/                     ##
## @Developed for: BME 494/598 Applied Programming      ##
##   (https://plaisierlab.engineering.asu.edu/)         ##
##   Arizona State University                           ##
##   242 ISTB1, 550 E Orange St                         ##
##   Tempe, AZ  85281                                   ##
## @Author:  Chris Plaisier                             ##
## @License:  GNU GPLv3                                 ##
##                                                      ##
## If this program is used in your analysis please      ##
## mention who built it. Thanks. :-)                    ##
##########################################################

## Your name
# Jessica Miron 1224633969




## Import required packages
import csv


## 1. Install chemparse
#  Deliverables:
#     a. In console window run:
#        pip install chemparse
#     b. Restart console:  Consoles tab -> Restart Kernel
#     c. Import 'chemparse' as 'cp'
#     d. Test chemparse using 'C6H12O6'
import chemparse as cp
glucose = cp.parse_formula('C6H12O6')
print(glucose)


## 2. (10pts) Load up 'inclassactivity_3_2_1.py' data and function
#  Deliverables:
#     a. Import your 'inclassactivity_3_2_1' as 'ica321' so you can use the 'molecular_weight' function
#        - Will have to be the full name of the file without the '.py':
import inclassactivity_3_2_1 as ica321
#     b. Test whether the imported 'molecular_weight' function works by printing out the molecular weight of glucose 'C6H12O6'
print(ica321.molecular_weight(glucose, ica321.ptoe))


## 3. (10pts) Read in molecules from 'molecules.txt', a file that has the columns molecule name and
#     molecular formula. Store into a dictionary with the molecule name as keys and molecular
#     formula as the values.
# Deliverables:
#   a. Open the 'molecules.txt' file
#      - Hint:  might be helpful to look at the file in a text editor first! Delimiter?
molecules = {}
with open('molecules.txt', 'r') as file:
    next(file)
    for row in file:
        parts = row.strip().split('\t')
        name, formula = parts
        name = name.casefold()
        molecules[name] = formula
#print(molecules)
#   b. Get rid of header
#   c. Iteratively split apart the contents of each line
#   d. Store into 'molecules' dictionary
#      - keys will be molecule name
#      - values will be molecular formula
#   e. Use a print statement to check everything worked as expected
print('\nMOLECULES\n')
print(molecules)




## 4. (20pts) Compute molecular weights
# Deliverables:
#   a. Iterate over the molecules
#   b. Compute the molecular_weight of each molecule
#   c. Store into 'molecular_weights' dictionary
#      - keys will be molecule name
#      - values will be molecular weight
#      - You must use Worksheet #9 molecular_weight and chemparse parse_formula functions
#   e. Use a print statement to check everything worked as expected
molecular_weights = {}
for molecule in molecules:
    moledict = cp.parse_formula(molecules[molecule])
    molecular_weights[molecule] = ica321.molecular_weight(moledict, ica321.ptoe)
print('\nMOLECULAR WEIGHTS\n')
print(molecular_weights)
print('\n')


## 5. (10pts) Read in DrugBank drug names and IDs from 'DrugBank.csv', a file that has DrugBank drug
#     names and unique computer readable DrugBank IDs. Store into a dictionary 'drugs' with the
#     drug names as keys and drug ID as values.
# Deliverables:
#   a. Open the 'DrugBank.csv' file
#      - Hint:  might be helpful to look at the file in a text editor first! Delimiter?
#   b. Get rid of header
#   c. Iteratively split apart the contents of each line
#   d. Store into 'drugs' dictionary
#      - keys will be drug name
#      - values will be drug ID
#   e. Use a print statement to check everything worked as expected
drugs = {}
with open('DrugBank.csv','r',newline = '') as file:
    next(file)
    reader = csv.reader(file)
    for row in reader:
        drug_id, drug_name = row
        drug_name = drug_name.casefold()
        drugs[drug_name] = drug_id
print('\nDRUGS\n')
print(drugs)



## 6. (20pts) Determine which drugs have a molecular weight in ‘molecular_weights’ and store into the
#     dictionary ‘drug_weights’.
# Deliverables:
#   a. Iterate over drug names
#   b. Check if the drug name exists in 'molecules'
#   c. If drug name in 'molecules' then add to 'drug_weights' dictionary
#      - keys will be drug name
#      - values will be the drugs molecular weight
#   d. Use a print statement to check everything worked as expected
drug_weights = {}
for drug_name in drugs:
    if drug_name in molecules:
        drug_weights[drug_name] = molecular_weights[drug_name]
print('\nDRUG WEIGHTS\n')
print(drug_weights)


## 7. (20pts) Identify drugs with molecular weight within 400 +/- 5 g/mol.
# Deliverables:
#   a. Iterate through drugs with molecular weights ('drug_weights')
#   b. Check if drug molecular weight is within range (400 +/- 5 g/mol)
#   c. If passes these filters then add drug to possible_drugs
#   d. Use a print statement to check everything worked as expected
possible_drugs = []
for drug in drug_weights:
    if 395 < float(drug_weights[drug]) < 405:
        possible_drugs.append(drug)
print('\nPOSSIBLE DRUGS\n')
print(possible_drugs)


## 8. (10pts) Write out CSV file with all possible drug and target combiniations.
#   a. The file that should have the following header:
#        Drug,DrugBank_ID,MW
#    Description of columns:
#      Drug = Drug name
#      DrugBank_ID = ID from DrugBank in 'drugs'
#      MW = molecular weight from 'drug_weights'
#
# Deliverables:
#   a. Iterate through 'possible_drugs'
#   c. Add information to to write out into a variable temporary variable 'tmp1'
#   d. Join information by commas and append to variable 'write_me'
#   e. After iterations, open file 'possible_drugs.csv' to write out
#   f. Write out 'write_me' joined by newlines
#   g. Open in a spreadsheet program to ensure all data is included that was requested
write_me = []
header = ['Drug','DrugBank_ID','MW']
write_me.append(header)
for drug in possible_drugs:
    tmp1 = [drug, drugs[drug], str(drug_weights[drug])]
    write_me.append(tmp1)
print(write_me)
with open('possible_drugs.csv','w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(write_me)






