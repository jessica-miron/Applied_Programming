##########################################################
## BME 494/598:  In-class activity 3.2 #1               ##
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
import json


## 1. Load up periodic table of elements data
# Deliverables:
#   a. Open the 'periodic_table_lookup.json' file
#   b. Load periodic table of elements data into variable called 'ptoe'
with open('periodic_table_lookup.json', 'r') as json_file:
    ptoe_all = json.load(json_file)
#   c. Print the keys for ptoe_all
#print(ptoe_all.keys())
#   d. Print the key 'order'
#print(ptoe_all['order'])
#   e. Print the key 'carbon'
#print(ptoe_all['carbon'])
#   f. Print the atomic symbol for 'carbon'
#print(ptoe_all['carbon']['symbol'])
#   g. Print the atomic mass for 'carbon'
#print(ptoe_all['carbon']['atomic_mass'])





## 2. (5pts) Build a dictionary that is keyed off atomic symobl and has atomic mass as a value
#     Molecules will be given as dictionaries with atomic symbol as key and value being the
#     number of atoms. The ptoe_all is keyed off the full atom name, and we want to have the
#     atomic mass keyed off the atomic symbol. {'C': 12.011, ... }
# Deliverables:
#   a. Initialize a new dictionary 'ptoe'
ptoe = {}
#   b. Iterate through all 'ptoe_all' atoms (easiest to use ptoe_all['order'] as the iterable)
#   c. Add each atom into 'ptoe' with atomic symbol ('symbol') as the key and atomic mass
#      ('atomic_mass') as the value
for atom in ptoe_all['order']:
    ptoe[ptoe_all[atom]['symbol']] = ptoe_all[atom]['atomic_mass']
#   d. Print the atomic mass of 'C' using 'ptoe', and check if equals 12.011
#print(ptoe['C'])
#   e. Print the atomic mass of 'Fe' using 'ptoe', and check if equals 55.8452
#print(ptoe['Fe'])
#   f. Print the atomic mass of 'U' using 'ptoe', and check if equals 238.028913
#print(ptoe['U'])



## 3. (5pts) Define a function 'molecular_weight' to find the molecular weight for molecules
# Deliverables:
#   a. Write a function 'molecular_weight' that:
#      - Takes two arguments 'molecule' and 'ptoe'
#        a. 'molecule' argument is a dictionary of atoms in the molecule with the value being the number of atoms
#        b. 'ptoe' the dictionary you developed above with atomic symbol as keys and atomic mass as values
#      - Use the atomic weights and the molecular composition to compute the molecular weight
#      - Return the molecular weight
#      - Docstrings to document the function
def molecular_weight(molecule,ptoe):
    """Calculate the molecular weight of a molecule
    Parameters
    ----------
    molecule : dict, molecule to find molecular weight of, format {'C':3,...}
    ptoe : dict, dictionary of atoms and atomic mass, format {'C':12.011,...}
    ----------
    Returns
    -------
    weight : float, molecular weight of molecule
    -------
    Notes
    -----
    Iterates through the given molecule to sum all the mulitplication of atoms
    by their atomic mass
    """
    weight = 0
    for atom in molecule.keys():
        weight = weight + (ptoe[atom] * molecule[atom])
    return weight
#   b. Print the results of these tests using the following molecules:
#      - Glycine = C2H5NO2
glycine = {'C':2,'H':5,'N':1,'O':2}
#print(molecular_weight(glycine,ptoe))
#      - Glucose = C6H12O6
glucose = {'C':6,'H':12,'O':6}
#print(molecular_weight(glucose,ptoe))
#      - Palmitic acid = C16H32O2
palmitic_acid = {'C':16,'H':32,'O':2}
#print(molecular_weight(palmitic_acid,ptoe))
#      - ATP = C10H16N5O13P3
atp = {'C':10,'H':16,'N':5,'O':13,'P':3}
#print(molecular_weight(atp,ptoe))
#      - Dichlorodifluoromethane = CCl2F2
dichlorodifluoromethane = {'C':1,'Cl':2,'F':2}
#print(molecular_weight(dichlorodifluoromethane,ptoe))
#      - Selenocysteine = C3H7NO2Se
selenocysteine = {'C':3,'H':7,'N':1,'O':2,'Se':1}
#print(molecular_weight(selenocysteine,ptoe))
#      - Heme B = C34H32O4N4Fe
heme_b = {'C':34,'H':32,'O':4,'N':4,'Fe':1}
#print(molecular_weight(heme_b,ptoe))

#   c. Check molecular weights on the web to make sure your code, function, and tests are working correctly

