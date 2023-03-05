"""
Simple demonstration of CQED-CIS method on MgH+ diatomic with
a bondlength of 2.2 Angstroms coupled to a photon with energy 4.75 eV.
This photon energy is chosen to be in resonance with the |X> -> |A>
(ground to first singlet excited-state) transition at the bondlength of 2.2 
Angstroms.  Three calculations will be performed:

1. Electric field vector has 0 magnitude (\lambda = (0, 0, 0)) and 0 energy
   allowing direct comparison to ordinary CIS

2. Electric field vector is z-polarized with magnitude 0.0125 atomic units 
   (\lambda = (0, 0, 0.125) and photon has energy 4.75 eV, which will split
   the eigenvectors proportional to the field strength, allowing comparison to 
   results in Figure 3 (top) in [McTague:2021:ChemRxiv] 

3. Electric field vector is z-polarized (\lambda = (0, 0, 0.0125)) and photon has complex
   energy 4.75 - 0.22i eV, allowing comparison to results in Figure 3 (bottom) in [McTague:2021:ChemRxiv]


"""

__authors__ = ["Jon McTague", "Jonathan Foley"]
__credits__ = ["Jon McTague", "Jonathan Foley"]

__copyright_amp__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2021-08-19"

# ==> Import Psi4, numpy, and helper_CS_CQED_CIS <==
import psi4
import numpy as np
from helper_ciss_prism import *
from psi4.driver.procrouting.response.scf_response import tdscf_excitations

# Set Psi4 & NumPy Memory Options
psi4.set_memory("2 GB")
psi4.core.set_output_file("output.dat", False)

numpy_memory = 2

#mol_str = """
#O
#H 1 1.1
#H 1 1.1 2 104
#symmetry c1
#"""

options_dict = {
'basis': 'sto-3g',
                  'scf_type': 'pk',
                  'e_convergence': 1e-10,
                  'd_convergence': 1e-10,
                  'save_jk' : True,
}
mol_str = """
Li
H 1 1.0
symmetry c1
"""

#mol_str = """
#Mg
#H 1 2.2
#symmetry c1
#1 1
#"""

# options dict
#options_dict = {
#    "basis": "cc-pVDZ",
#    "save_jk": True,
#    "scf_type": "pk",
#    "e_convergence": 1e-10,
#    "d_convergence": 1e-10,
#}


# set psi4 options and geometry
psi4.set_options(options_dict)
mol = psi4.geometry(mol_str)

om_1 = 4.75
lam_1 = np.array([0.0, 0.0, 0.0])

om_2 = 4.75 / psi4.constants.Hartree_energy_in_eV 
lam_2 = np.array([0.01, 0.01, 0.0125])
#lam_2 = np.array([0.1, 0.1, 0.1])

# run cs_cqed_cis() for the three cases
cqed_cis_1 = cs_cqed_cis(lam_1, om_1, mol_str, options_dict)
cqed_cis_2 = cs_cqed_cis(lam_2, om_2, mol_str, options_dict)


cqed_cis_e_1 = cqed_cis_1["CISS-PF ENERGY"]
scf_e_1 = cqed_cis_1["CQED-RHF ENERGY"]

cqed_cis_e_2 = cqed_cis_2["CISS-PF ENERGY"]
scf_e_2 = cqed_cis_1["CQED-RHF ENERGY"]


print(
    "    CASE 1 CQED-CIS LOWEST EXCITATION ENERGY (eV)   %.4f"
    % (np.real(cqed_cis_e_1[1]) * psi4.constants.Hartree_energy_in_eV)
)

print(
    "\n    PRINTING RESULTS FOR CASE 2: HBAR * OMEGA = 4.75 eV, LAMBDA = (0, 0, 0.0125) A.U."
)

print("CORRELATION:", np.real(cqed_cis_e_2[0]) * psi4.constants.Hartree_energy_in_eV)
print("|G> (Hartrees)  %.8f" % (scf_e_2 + np.real(cqed_cis_e_2[0]))) 
print(
    "\n    CASE 2 |X,0> -> |LP> Energy (eV)                %.8f"
    % (np.real(cqed_cis_e_2[1]) * psi4.constants.Hartree_energy_in_eV)
)
print(
    "    CASE 2 |X,0> -> |UP> Energy (eV)                %.8f"
    % (np.real(cqed_cis_e_2[2]) * psi4.constants.Hartree_energy_in_eV)
)


# check to see that the CQED-RHF energy matches ordinary RHF energy for case 1
###psi4.compare_values(psi4_rhf_e, scf_e_1, 8, "CASE 1 SCF E")

# check to see if first CQED-CIS excitation energy matches first CIS excitation energy for case 1
###psi4.compare_values(cqed_cis_e_1[1], psi4_excitation_e[0], 8, "CASE 1 CQED-CIS E")


# check to see if first CQED-CIS excitation energy matches value from [McTague:2021:ChemRxiv] Figure 3 for case 2
# This still needs to be corrected in the paper!
###psi4.compare_values(cqed_cis_e_2[1], 0.1655708380, 8, "CASE 2 CQED-CIS E")
