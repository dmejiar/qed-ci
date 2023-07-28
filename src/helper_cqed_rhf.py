"""
Helper function for CQED_RHF
References:
    Equations and algorithms from 
    [Haugland:2020:041043], [DePrince:2021:094112], and [McTague:2021:ChemRxiv] 
    JJF Note: This implementation utilizes only electronic dipole contributions 
    and ignore superflous nuclear dipole terms!
"""

__authors__ = ["Jon McTague", "Jonathan Foley"]
__credits__ = ["Jon McTague", "Jonathan Foley"]

__copyright_amp__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2021-08-19"

# ==> Import Psi4, NumPy, & SciPy <==
import psi4
import numpy as np

def cqed_rhf(lambda_vector, molecule_string, psi4_options_dict):
    """Computes the QED-RHF energy and density
    Arguments
    ---------
    lambda_vector : 1 x 3 array of floats
        the electric field vector, see e.g. Eq. (1) in [DePrince:2021:094112]
        and (15) in [Haugland:2020:041043]
    molecule_string : string
        specifies the molecular geometry
    options_dict : dictionary
        specifies the psi4 options to be used in running the canonical RHF
    Returns
    -------
    cqed_rhf_dictionary : dictionary
        Contains important quantities from the cqed_rhf calculation, with keys including:
            'RHF ENERGY' -> result of canonical RHF calculation using psi4 defined by molecule_string and psi4_options_dict
            'CQED-RHF ENERGY' -> result of CQED-RHF calculation, see Eq. (13) of [McTague:2021:ChemRxiv]
            'CQED-RHF C' -> orbitals resulting from CQED-RHF calculation
            'CQED-RHF DENSITY MATRIX' -> density matrix resulting from CQED-RHF calculation
            'CQED-RHF EPS'  -> orbital energies from CQED-RHF calculation
            'PSI4 WFN' -> wavefunction object from psi4 canonical RHF calcluation
            'CQED-RHF DIPOLE MOMENT' -> total dipole moment from CQED-RHF calculation (1x3 numpy array)
            'NUCLEAR DIPOLE MOMENT' -> nuclear dipole moment (1x3 numpy array)
            'NUCLEAR REPULSION ENERGY' -> Total nuclear repulsion energy
    Example
    -------
    >>> cqed_rhf_dictionary = cqed_rhf([0., 0., 1e-2], '''\nMg\nH 1 1.7\nsymmetry c1\n1 1\n''', psi4_options_dictionary)
    """
    # define geometry using the molecule_string
    mol = psi4.geometry(molecule_string)
    # define options for the calculation
    psi4.set_options(psi4_options_dict)
    # run psi4 to get ordinary scf energy and wavefunction object
    psi4_rhf_energy, wfn = psi4.energy("scf", return_wfn=True)

    # Create instance of MintsHelper class
    mints = psi4.core.MintsHelper(wfn.basisset())

    # Grab data from wavfunction
    # number of doubly occupied orbitals
    ndocc = wfn.nalpha()

    # grab all transformation vectors and store to a numpy array
    C = np.asarray(wfn.Ca())

    # use canonical RHF orbitals for guess CQED-RHF orbitals
    Cocc = C[:, :ndocc]

    # form guess density
    D = np.einsum("pi,qi->pq", Cocc, Cocc)  # [Szabo:1996] Eqn. 3.145, pp. 139

    # Integrals required for CQED-RHF
    # Ordinary integrals first
    V_ao = np.asarray(mints.ao_potential())
    T_ao = np.asarray(mints.ao_kinetic())

    # Extra terms for Pauli-Fierz Hamiltonian
    # electronic dipole integrals in AO basis
    mu_ao_x = np.asarray(mints.ao_dipole()[0])
    mu_ao_y = np.asarray(mints.ao_dipole()[1])
    mu_ao_z = np.asarray(mints.ao_dipole()[2])

    # \lambda \cdot \mu_el (see within the sum of line 3 of Eq. (9) in [McTague:2021:ChemRxiv])
    d_el_ao = lambda_vector[0] * mu_ao_x
    d_el_ao += lambda_vector[1] * mu_ao_y
    d_el_ao += lambda_vector[2] * mu_ao_z

    # compute electronic dipole expectation value with
    # canonincal RHF density
    mu_exp_x = np.einsum("pq,pq->", 2 * mu_ao_x, D)
    mu_exp_y = np.einsum("pq,pq->", 2 * mu_ao_y, D)
    mu_exp_z = np.einsum("pq,pq->", 2 * mu_ao_z, D)

    # get electronic dipole expectation value
    mu_exp_el = np.array([mu_exp_x, mu_exp_y, mu_exp_z])

    # get nuclear dipole moment
    mu_nuc = np.array(
        [mol.nuclear_dipole()[0], mol.nuclear_dipole()[1], mol.nuclear_dipole()[2]]
    )
    rhf_dipole_moment = mu_exp_el + mu_nuc
    # We need to carry around the electric field dotted into the nuclear dipole moment

    # \lambda_vecto \cdot < \mu > where <\mu> contains ONLY electronic contributions
    d_nuc = np.dot(lambda_vector, mu_nuc)

    # quadrupole arrays
    Q_ao_xx = np.asarray(mints.ao_quadrupole()[0])
    Q_ao_xy = np.asarray(mints.ao_quadrupole()[1])
    Q_ao_xz = np.asarray(mints.ao_quadrupole()[2])
    Q_ao_yy = np.asarray(mints.ao_quadrupole()[3])
    Q_ao_yz = np.asarray(mints.ao_quadrupole()[4])
    Q_ao_zz = np.asarray(mints.ao_quadrupole()[5])

    # Pauli-Fierz 1-e quadrupole terms, Line 2 of Eq. (9) in [McTague:2021:ChemRxiv]
    Q_ao = -0.5 * lambda_vector[0] * lambda_vector[0] * Q_ao_xx
    Q_ao -= 0.5 * lambda_vector[1] * lambda_vector[1] * Q_ao_yy
    Q_ao -= 0.5 * lambda_vector[2] * lambda_vector[2] * Q_ao_zz

    # accounting for the fact that Q_ij = Q_ji
    # by weighting Q_ij x 2 which cancels factor of 1/2
    Q_ao -= lambda_vector[0] * lambda_vector[1] * Q_ao_xy
    Q_ao -= lambda_vector[0] * lambda_vector[2] * Q_ao_xz
    Q_ao -= lambda_vector[1] * lambda_vector[2] * Q_ao_yz

    # Pauli-Fierz 1-e dipole terms scaled <\mu>_e
    d_nuc_d_el_ao = d_nuc * d_el_ao

    # Pauli-Fierz (\lambda \cdot <\mu>_e ) ^ 2
    d_nuc_sq = 0.5 * d_nuc ** 2

    
    cqed_rhf_dict = {
        "RHF ENERGY": psi4_rhf_energy,
        "CQED-RHF C": C,
        "CQED-RHF DENSITY MATRIX": D,
        "CQED-RHF EPS": e,
        "PSI4 WFN": wfn,
        "CQED-RHF ELECTRONIC DIPOLE MOMENT": mu_exp_el,
        "NUCLEAR DIPOLE MOMENT": mu_nuc,
        "CQED-RHF DIPOLE MOMENT": mu_exp_el + mu_nuc,
        "RHF DIPOLE MOMENT": rhf_dipole_moment,
        "NUCLEAR REPULSION ENERGY": Enuc,
        "DIPOLE AO X": mu_ao_x,
        "DIPOLE AO Y": mu_ao_y,
        "DIPOLE AO Z": mu_ao_z,
        "QUADRUPOLE AO XX": Q_ao_xx,
        "QUADRUPOLE AO YY": Q_ao_yy,
        "QUADRUPOLE AO ZZ": Q_ao_zz,
        # the cross-terms are not scaled by 2 here
        "QUADRUPOLE AO XY": Q_ao_xy,
        "QUADRUPOLE AO XZ": Q_ao_xz,
        "QUADRUPOLE AO YZ": Q_ao_zz,
        "NUCLEAR DIPOLE ENERGY": d_nuc_sq,
        "PF 1-E DIPOLE MATRIX AO": d_el_ao,
        "PF 1-E QUADRUPOLE MATRIX AO": Q_ao,
        "PF 1-E SCALED DIPOLE MATRIX" : d_nuc_d_el_ao,
        "1-E KINETIC MATRIX AO" : T_ao,
        "1-E POTENTIAL MATRIX AO": V_ao
    }
    return cqed_rhf_dict
