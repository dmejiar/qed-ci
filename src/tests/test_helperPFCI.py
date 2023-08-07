import psi4
from helper_PFCI import PFHamiltonianGenerator
from helper_PFCI import Determinant
from helper_cqed_rhf import cqed_rhf
import numpy as np
import pytest
import sys

np.set_printoptions(threshold=sys.maxsize)


def test_h2o_qed_fci_no_cavity():

    options_dict = {
        "basis": "sto-3g",
        "scf_type": "pk",
        "e_convergence": 1e-10,
        "d_convergence": 1e-10,
    }

    cavity_dict = {
        'omega_value' : 0.0,
        'lambda_vector' : np.array([0, 0, 0]),
        'ci_level' : 'fci',
        'number_of_photons' : 1,
        'davidson_roots' : 4,
        'davidson_threshold' : 1e-8
    }

    # molecule string for H2O
    h2o_string = """
    O
    H 1 1.1
    H 1 1.1 2 104
    symmetry c1
    """

    test_pf = PFHamiltonianGenerator(
        h2o_string,
        options_dict,
        cavity_dict
    )
    expected_g   = -75.0129801827
    excpected_e1 = -74.7364625844

    actual_g = test_pf.CIeigs[0] # <== ground state
    actual_e1 = test_pf.CIeigs[2] # <== first excited state

    assert np.isclose(actual_g, expected_g)
    assert np.isclose(actual_e1, excpected_e1)


def test_mghp_qed_cis_no_cavity():
    # options for mgf
    mol_str = """
    Mg
    H 1 2.2
    symmetry c1
    1 1
    """

    options_dict = {
        "basis": "cc-pVDZ",
        "scf_type": "pk",
        "e_convergence": 1e-10,
        "d_convergence": 1e-10,
    }

    cavity_dict = {
        'omega_value' : 0.0,
        'lambda_vector' : np.array([0, 0, 0]),
        'ci_level' : 'cis',
        'davidson_roots' : 8,
        'davidson_threshold' : 1e-8
    }

    mol = psi4.geometry(mol_str)

    psi4.set_options(options_dict)

    #energy from psi4numpy
    expected_mghp_eg = -199.8639591041915
    
    expected_mghp_e1 = -199.6901102832973

    test_pf = PFHamiltonianGenerator(
        mol_str,
        options_dict,
        cavity_dict
    )

    #e_fci, wavefunctions = np.linalg.eigh(test_pf.H_PF)
    actual_e0 = test_pf.CIeigs[0] # <== ground state
    actual_e1 = test_pf.CIeigs[4] # <== root 5 is first singlet excited state

    assert np.isclose(actual_e0, expected_mghp_eg)
    assert np.isclose(actual_e1, expected_mghp_e1)

def test_mghp_qed_cis_with_cavity():
    # options for mgf
    mol_str = """
    Mg
    H 1 2.2
    symmetry c1
    1 1
    """

    options_dict = {
        "basis": "cc-pVDZ",
        "scf_type": "pk",
        "e_convergence": 1e-10,
        "d_convergence": 1e-10,
    }

    cavity_dict = {
        'omega_value' : 4.75 / psi4.constants.Hartree_energy_in_eV,
        'lambda_vector' : np.array([0, 0, 0.0125]),
        'ci_level' : 'cis',
        'davidson_roots' : 8,
        'davidson_threshold' : 1e-8
    }

    mol = psi4.geometry(mol_str)

    psi4.set_options(options_dict)

    #energy from psi4numpy
    expected_mghp_g_e = -199.86358254419457
    expected_mghp_lp_e = -199.69776087489558
    expected_mghp_up_e = -199.68066502792058


    test_pf = PFHamiltonianGenerator(
        mol_str,
        options_dict,
        cavity_dict
    )

    #e_fci, wavefunctions = np.linalg.eigh(test_pf.H_PF)
    actual_g = test_pf.CIeigs[0] # <== ground state
    actual_lp = test_pf.CIeigs[2] # <== root 3 is LP
    actual_up = test_pf.CIeigs[5] # <== root 6 is UP

    assert np.isclose(actual_g, expected_mghp_g_e)
    assert np.isclose(actual_lp, expected_mghp_lp_e)
    assert np.isclose(actual_up, expected_mghp_up_e)


def test_mghp_qed_cis_with_cavity_canonical_mo():
    # options for mgf
    mol_str = """
    Mg
    H 1 2.2
    symmetry c1
    1 1
    """

    options_dict = {
        "basis": "cc-pVDZ",
        "scf_type": "pk",
        "e_convergence": 1e-10,
        "d_convergence": 1e-10,
    }

    cavity_dict = {
        'omega_value' : 4.75 / psi4.constants.Hartree_energy_in_eV,
        'lambda_vector' : np.array([0, 0, 0.0125]),
        'ci_level' : 'cis',
        'full_diagonalization' : True,
        'canonical_mos' : True
    }

    mol = psi4.geometry(mol_str)

    psi4.set_options(options_dict)

    #energy from psi4numpy
    expected_mghp_g_e = -199.86358254419457
    expected_mghp_lp_e = -199.69776087489558
    expected_mghp_up_e = -199.68066502792058

    test_pf = PFHamiltonianGenerator(
        mol_str,
        options_dict,
        cavity_dict
    )

    #e_fci, wavefunctions = np.linalg.eigh(test_pf.H_PF)
    actual_g = test_pf.CIeigs[0] # <== ground state
    actual_lp = test_pf.CIeigs[2] # <== root 3 is LP
    actual_up = test_pf.CIeigs[5] # <== root 6 is UP

    assert np.isclose(actual_g, expected_mghp_g_e )
    assert np.isclose(actual_lp, expected_mghp_lp_e)
    assert np.isclose(actual_up, expected_mghp_up_e)

def test_ch2o_qed_cas_88_no_cavity():

    options_dict = {
        "basis": "cc-pVDZ",
        "scf_type": "pk",
        "e_convergence": 1e-10,
        "d_convergence": 1e-10,
    }

    cavity_dict = {
        'omega_value' : 0.0,
        'lambda_vector' : np.array([0, 0, 0]),
        'ci_level' : 'cas',
        'number_of_photons' : 0,
        'nact_els' : 8,
        'nact_orbs' : 8,
        'full_diagonalization' : True
    }

    # molecule string for H2O
    ch2o_string = """
    0 1
    O
    C             1    1.2448979591836735
    H             2    1.120350      1  122.478805
    H             2    1.120350      1  122.478805      3  180.000000
    symmetry c1
    """

    test_pf = PFHamiltonianGenerator(
        ch2o_string,
        options_dict,
        cavity_dict
    )
    
    # expected eigenvalues from canonical_mo_option commit 36377939e45c5bcf9d1b6c8d2ecdd6dc29e8ecdd
    expected_e = np.array([-113.9023426,  -113.75614494, -113.7441367,  -113.69468587, -113.58155048])

    actual_e = test_pf.CIeigs[:5] # <== ground state

    assert np.allclose(expected_e, actual_e)


