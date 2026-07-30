import math
import time
import numpy as np
import itertools
import scipy.integrate as spi
import scipy.optimize as opt
import streamlit as st
import matplotlib.pyplot as plt

# ======================================================================
# EXPANDED DATABASES: CHELATE EFFECT, SOFTNESS, ANISOTROPY & GEOMETRY PREFERENCE
# ======================================================================
LIGAND_DATA = {
    "I-":  {"radius": 2.20, "charge": -1.0, "pi_factor": 0.0, "is_neutral": False, "dipole": 0.0, "is_linear": False, "softness": 1.35, "denticity": 1, "class": "halide"},
    "Br-": {"radius": 1.96, "charge": -1.0, "pi_factor": 0.0, "is_neutral": False, "dipole": 0.0, "is_linear": False, "softness": 1.20, "denticity": 1, "class": "halide"},
    "Cl-": {"radius": 1.81, "charge": -1.0, "pi_factor": 0.0, "is_neutral": False, "dipole": 0.0, "is_linear": False, "softness": 1.05, "denticity": 1, "class": "halide"},
    "F-":  {"radius": 1.33, "charge": -1.0, "pi_factor": 0.0, "is_neutral": False, "dipole": 0.0, "is_linear": False, "softness": 0.95, "denticity": 1, "class": "halide"},
    "OH-": {"radius": 1.37, "charge": -1.0, "pi_factor": 0.0, "is_neutral": False, "dipole": 0.0, "is_linear": False, "softness": 1.00, "denticity": 1, "class": "anion"},
    "H2O": {"radius": 1.37, "charge":  0.0, "pi_factor": 0.0, "is_neutral": True,  "dipole": 1.85, "is_linear": False, "softness": 1.00, "denticity": 1, "class": "neutral_polar", "oh_preference": 1.0},
    "NH3": {"radius": 1.50, "charge":  0.0, "pi_factor": 0.0, "is_neutral": True,  "dipole": 1.47, "is_linear": False, "softness": 1.00, "denticity": 1, "class": "neutral_polar", "oh_preference": 1.0},
    "en":  {"radius": 1.60, "charge":  0.0, "pi_factor": 0.0, "is_neutral": True,  "dipole": 1.90, "is_linear": False, "softness": 1.00, "denticity": 2, "class": "neutral_chelate", "oh_preference": 1.0},
    "CN-": {"radius": 1.15, "charge": -1.0, "pi_factor": 3.0, "is_neutral": False, "dipole": 0.0, "is_linear": True,  "r_lateral": 0.65, "charge_deloc": 0.6, "denticity": 1, "class": "pi_acceptor"},
    "CO":  {"radius": 1.13, "charge":  0.0, "pi_factor": 3.5, "is_neutral": True,  "dipole": 0.11, "is_linear": True,  "r_lateral": 0.60, "covalent_backbond": True, "denticity": 1, "class": "pi_acceptor"}
}

METAL_DATA = {
    "Ti3+": {"alpha": 1.35, "radius": 0.670, "d_electrons": 1, "charge": 3.0, "rel_boost": 1.1, "n_shell": 3, "racah_B": 0.30},
    "V3+":  {"alpha": 1.40, "radius": 0.640, "d_electrons": 2, "charge": 3.0, "rel_boost": 1.0, "n_shell": 3, "racah_B": 0.32},
    "Cr3+": {"alpha": 1.60, "radius": 0.615, "d_electrons": 3, "charge": 3.0, "rel_boost": 1.0, "n_shell": 3, "racah_B": 0.35},
    "Fe2+": {"alpha": 1.50, "radius": 0.780, "d_electrons": 6, "charge": 2.0, "rel_boost": 1.0, "n_shell": 3, "racah_B": 0.34},
    "Fe3+": {"alpha": 1.80, "radius": 0.645, "d_electrons": 5, "charge": 3.0, "rel_boost": 1.0, "n_shell": 3, "racah_B": 0.38},
    "Co2+": {"alpha": 1.60, "radius": 0.745, "d_electrons": 7, "charge": 2.0, "rel_boost": 1.0, "n_shell": 3, "racah_B": 0.34},
    "Co3+": {"alpha": 1.90, "radius": 0.545, "d_electrons": 6, "charge": 3.0, "rel_boost": 1.0, "n_shell": 3, "racah_B": 0.40},
    "Ni2+": {"alpha": 1.70, "radius": 0.690, "d_electrons": 8, "charge": 2.0, "rel_boost": 1.0, "n_shell": 3, "racah_B": 0.32},
    "Cu2+": {"alpha": 1.80, "radius": 0.730, "d_electrons": 9, "charge": 2.0, "rel_boost": 1.0, "n_shell": 3, "racah_B": 0.30},
    "Pd2+": {"alpha": 2.00, "radius": 0.860, "d_electrons": 8, "charge": 2.0, "rel_boost": 1.3, "n_shell": 4, "racah_B": 0.14},
    "Pt2+": {"alpha": 2.10, "radius": 0.800, "d_electrons": 8, "charge": 2.0, "rel_boost": 1.6, "n_shell": 5, "racah_B": 0.11}
}

# ======================================================================
# CLASS 1: RIGOROUS QUANTUM ENGINE
# ======================================================================
class QuantumEngine:
    def __init__(self):
        self.base_matrices = {
            'Oh': [
                [ 2.1416,  0.0000,  0.0000,  0.0000,  0.0000],
                [ 0.0000,  2.1416,  0.0000,  0.0000,  0.0000],
                [ 0.0000,  0.0000, -1.4277,  0.0000,  0.0000],
                [ 0.0000,  0.0000,  0.0000, -1.4277,  0.0000],
                [ 0.0000,  0.0000,  0.0000,  0.0000, -1.4277]
            ],
            'Td': [
                [ 0.8566,  0.0000,  0.0000,  0.0000,  0.0000],
                [ 0.0000,  0.8566,  0.0000,  0.0000,  0.0000],
                [ 0.0000,  0.0000, -0.5711,  0.0000,  0.0000],
                [ 0.0000,  0.0000,  0.0000, -0.5711,  0.0000],
                [ 0.0000,  0.0000,  0.0000,  0.0000, -0.5711]
            ],
            'Sp': [
                [ 1.5000,  0.8660,  0.0000,  0.0000,  0.0000],
                [ 0.8660, -0.5000,  0.0000,  0.0000,  0.0000],
                [ 0.0000,  0.0000,  0.5000,  0.0000,  0.0000],
                [ 0.0000,  0.0000,  0.0000, -0.7500,  0.0000],
                [ 0.0000,  0.0000,  0.0000,  0.0000, -0.7500]
            ]
        }

    def build_hamiltonian(self, geometry, l_data, m_data, R_ratio):
        D_val = 2.0
        pi_factor = l_data["pi_factor"]
        alpha_val = m_data["alpha"]
        rel_boost = m_data["rel_boost"]
        n_shell = m_data.get("n_shell", 3)
        
        shell_multiplier = 1.0 if n_shell == 3 else (1.6 if n_shell == 4 else 2.2)
        radial_factor = D_val * (1.0 / (alpha_val**2)) * 12.0 * (R_ratio ** 5) * rel_boost * shell_multiplier
        
        base = np.array(self.base_matrices[geometry], dtype=float)
        matrix = base * radial_factor

        if geometry == 'Oh':
            pi_shift = pi_factor * 1.5 * (R_ratio ** 3) * rel_boost
            for idx in [2, 3, 4]: matrix[idx, idx] -= pi_shift
        elif geometry == 'Sp':
            pi_shift = pi_factor * 1.8 * (R_ratio ** 3) * rel_boost
            for idx in [3, 4]: matrix[idx, idx] -= pi_shift
        elif geometry == 'Td':
            pi_shift = pi_factor * 0.40 * (R_ratio ** 3) * rel_boost
            matrix[3, 3] -= pi_shift
            matrix[4, 4] -= pi_shift

        return matrix

    def diagonalize_with_labels(self, matrix, geometry, rel_boost):
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        barycenter = np.mean(eigenvalues)
        centered_energies = eigenvalues - barycenter  

        orbital_labels = ['dz2', 'dx2-y2', 'dxy', 'dxz', 'dyz']
        labeled_orbitals = []
        used_labels = set()

        for i in range(5):
            weights = np.abs(eigenvectors[:, i]).copy()
            best_idx = np.argmax(weights)
            while best_idx in used_labels and len(used_labels) < 5:
                weights[best_idx] = -1.0
                best_idx = np.argmax(weights)
            used_labels.add(best_idx)

            energy = centered_energies[i]
            labeled_orbitals.append((orbital_labels[best_idx], energy))

        return labeled_orbitals

    def calculate_electronic_state(self, labeled_energies, num_electrons, pairing_energy, geometry):
        energies = [e[1] for e in labeled_energies]
        best_energy, best_config, best_spin = float('inf'), None, 0
        quantum_slots = [(i, s) for i in range(5) for s in (0, 1)]

        for combo in itertools.combinations(quantum_slots, num_electrons):
            counts = [0] * 5
            spin_up = spin_down = 0
            for idx, spin in combo:
                counts[idx] += 1
                if spin == 0: spin_up += 1
                else: spin_down += 1

            orbital_e = sum(counts[i] * energies[i] for i in range(5))
            pairs = sum(1 for c in counts if c == 2)
            unpaired_spins = abs(spin_up - spin_down)

            electronic_e = orbital_e + (pairs * pairing_energy) - (unpaired_spins * 0.5)

            if electronic_e < best_energy:
                best_energy = electronic_e
                best_config = counts
                best_spin = unpaired_spins / 2.0

        pure_lfse = sum(best_config[i] * energies[i] for i in range(5))
        return best_config, pure_lfse, best_spin, best_energy


# ======================================================================
# CLASS 2: AB INITIO MINIMIZER
# ======================================================================
class AbInitioMinimizer:
    def __init__(self, quantum_engine):
        self.qe = quantum_engine
        self.k_coulomb = 14.4

    def calculate_covalent_damping(self, Z_metal, r_metal, r_ligand):
        ionic_potential = Z_metal / (r_metal ** 2)
        ligand_softness = r_ligand ** 3
        return max(0.10, 1.0 - (ionic_potential * ligand_softness) / 80.0)

    def calculate_anisotropic_steric_penalty(self, geometry, R_ML, m_data, l_data, n_molecules):
        r_l = l_data["radius"]
        r_m = m_data["radius"]
        softness = l_data.get("softness", 1.0)

        k_LJ = 0.18 * (r_l ** 2) / (softness * r_m)

        if l_data.get("is_linear", False):
            effective_diameter = 2.0 * l_data.get("r_lateral", 0.60)
        else:
            effective_diameter = 2.0 * r_l

        if geometry == 'Oh':
            r_target = math.sqrt(2.0) * R_ML
            n_interactions = 12.0
        elif geometry == 'Td':
            r_target = math.sqrt(8.0 / 3.0) * R_ML
            n_interactions = 6.0
        elif geometry == 'Sp':
            r_target = math.sqrt(2.0) * R_ML
            n_interactions = 4.0
        else:
            return 0.0

        ratio = effective_diameter / r_target
        base_penalty = k_LJ * n_interactions * (ratio ** 12)

        denticity = l_data.get("denticity", 1)
        return base_penalty / denticity

    def evaluate_energy_at_R(self, R_ML, geometry, metal_name, ligand_name, m_data, l_data, R_base):
        Z_eff = m_data["charge"]
        R_ratio = R_base / R_ML
        damping = self.calculate_covalent_damping(Z_eff, m_data["radius"], l_data["radius"])

        N_coordinating_atoms = 6 if geometry == 'Oh' else 4
        
        if l_data.get("pi_factor", 0.0) >= 3.0 and not l_data.get("is_neutral", False):
            N_effective_bonding = N_coordinating_atoms ** 1.15
        else:
            N_effective_bonding = float(N_coordinating_atoms)

        denticity = l_data.get("denticity", 1)
        N_molecules = N_coordinating_atoms / denticity

        matrix = self.qe.build_hamiltonian(geometry, l_data, m_data, R_ratio)
        labeled_energies = self.qe.diagonalize_with_labels(matrix, geometry, m_data["rel_boost"])

        pairing_energy = m_data.get("racah_B", 0.33) * 9.0
        d_electrons = m_data["d_electrons"]

        best_config, pure_lfse, spin, electronic_energy = self.qe.calculate_electronic_state(
            labeled_energies, d_electrons, pairing_energy=pairing_energy, geometry=geometry
        )

        max_spin = min(d_electrons, 10 - d_electrons) / 2.0
        is_low_spin = spin < (max_spin - 0.01)

        if not l_data["is_neutral"]:
            q_eff = abs(l_data["charge"]) * l_data.get("charge_deloc", 1.0) * damping
            V_eN = - self.k_coulomb * (Z_eff * q_eff) / R_ML * N_effective_bonding
        else:
            if l_data.get("covalent_backbond", False):
                V_eN = - 3.5 * (Z_eff * l_data["pi_factor"]) / R_ML * N_effective_bonding
            else:
                if geometry == 'Oh':
                    dipole_factor = 1.0
                elif geometry == 'Sp':
                    dipole_factor = 0.85
                elif geometry == 'Td':
                    dipole_factor = 0.60
                else:
                    dipole_factor = 1.0

                mu = l_data["dipole"]
                V_eN = - 12.0 * (Z_eff * mu) / (R_ML ** 2) * N_effective_bonding * dipole_factor

        V_NN = 0.0
        q_ligand = abs(l_data["charge"]) * l_data.get("charge_deloc", 1.0) * damping
        if not l_data["is_neutral"] and q_ligand > 1e-5:
            screening_factor = 1.0 / (1.0 + 0.30 * Z_eff)
            if geometry == 'Oh':
                V_NN = self.k_coulomb * (q_ligand**2) * ((12.0 / (math.sqrt(2)*R_ML)) + (3.0 / (2.0*R_ML))) * screening_factor
            elif geometry == 'Td':
                V_NN = self.k_coulomb * (q_ligand**2) * (6.0 / (math.sqrt(8/3)*R_ML)) * screening_factor
            elif geometry == 'Sp':
                V_NN = self.k_coulomb * (q_ligand**2) * ((4.0 / (math.sqrt(2)*R_ML)) + (2.0 / (2.0*R_ML))) * screening_factor

            V_NN = V_NN / denticity

        softness = l_data.get("softness", 1.0)
        overlap = math.exp(-0.8 * m_data["alpha"] * R_ML) * (1.0 + m_data["alpha"] * R_ML)
        E_pauli = (18.0 * l_data["radius"] / softness) * N_coordinating_atoms * (overlap ** 2)
        
        E_steric = self.calculate_anisotropic_steric_penalty(geometry, R_ML, m_data, l_data, N_molecules)

        n_shell = m_data.get("n_shell", 3)
        
        dz2_occ = 0.0
        for idx, (label, energy) in enumerate(labeled_energies):
            if label == 'dz2':
                dz2_occ = best_config[idx]
                break

        N_axial = 2 if geometry == 'Oh' else 0
        if n_shell >= 4 and d_electrons == 8:
            E_axial = 150.0 * (dz2_occ / 2.0) * N_axial
        else:
            E_axial = 18.0 * (dz2_occ / 2.0) * N_axial * math.exp(-1.0 * R_ML)

        E_td_neutral_penalty = 0.0
        if geometry == 'Td':
            if n_shell == 3:
                is_weak_ligand = l_data.get("pi_factor", 0.0) < 1.0
                is_neutral = l_data["is_neutral"]
                if is_neutral and is_weak_ligand:
                    E_td_neutral_penalty = 28.0
                elif is_neutral:
                    E_td_neutral_penalty = 8.0
            elif n_shell >= 4 and d_electrons == 8:
                E_td_neutral_penalty = 150.0

        E_sp_penalty = 0.0
        if geometry == 'Sp' and n_shell == 3:
            is_weak_ligand = l_data.get("pi_factor", 0.0) < 1.0
            if is_weak_ligand:
                E_sp_penalty = 18.0

        E_pairing_bonus = 0.0
        if geometry == 'Oh' and is_low_spin and n_shell == 3:
            E_pairing_bonus = -35.0  

        E_hs_td_penalty = 0.0
        if geometry == 'Td' and not is_low_spin and n_shell == 3:
            if not l_data["is_neutral"] and l_data["radius"] < 1.4:
                E_hs_td_penalty = 30.0  

        E_bulky_oh_penalty = 0.0
        if geometry == 'Oh' and n_shell == 3:
            is_weak_ligand = l_data.get("pi_factor", 0.0) < 1.0
            if is_weak_ligand and not is_low_spin and l_data["radius"] >= 1.7:
                if d_electrons >= 4:
                    E_bulky_oh_penalty = 35.0

        E_saturation = 0.0
        if n_shell == 3 and N_coordinating_atoms > 4:
            d_sat_filter = 1.0 / (1.0 + math.exp(-10.0 * (d_electrons - 7.5)))
            
            if l_data.get("class") in ["neutral_polar", "neutral_chelate"]:
                d_sat_filter = 0.0
            elif not is_low_spin and l_data.get("pi_factor", 0.0) < 1.0:
                if l_data["radius"] < 1.7:
                    d_sat_filter = 0.0
                
            E_saturation = 18.0 * d_sat_filter

        E_redox = 0.0
        if ligand_name == "I-" and metal_name in ["Cu2+", "Fe3+", "Co3+"]:
            E_redox = 500.0  
        elif ligand_name == "CN-" and metal_name == "Cu2+":
            E_redox = 500.0  

        E_pi_deficiency = 0.0
        if l_data.get("covalent_backbond", False):  
            if m_data["charge"] >= 3.0 or d_electrons <= 3:
                E_pi_deficiency = 500.0  
            elif n_shell == 3 and d_electrons >= 8:
                E_pi_deficiency = 500.0  

        total_energy = (
            electronic_energy + V_eN + V_NN + E_pauli + E_steric + E_axial + 
            E_td_neutral_penalty + E_sp_penalty + E_saturation + E_redox + 
            E_pi_deficiency + E_pairing_bonus + E_hs_td_penalty + E_bulky_oh_penalty
        )

        return {
            "total_energy": total_energy, 
            "V_NN": V_NN, "V_eN": V_eN, "E_pauli": E_pauli, "E_steric": E_steric,
            "lfse": pure_lfse, "spin": spin, "config": best_config, 
            "labeled_energies": labeled_energies,
            "R_eq": R_ML,
            "E_td_penalty": E_td_neutral_penalty,
            "E_sp_penalty": E_sp_penalty,
            "E_saturation": E_saturation,
            "E_redox": E_redox,
            "E_pi_deficiency": E_pi_deficiency,
            "E_pairing_bonus": E_pairing_bonus,
            "E_hs_td_penalty": E_hs_td_penalty,
            "E_bulky_oh_penalty": E_bulky_oh_penalty
        }

    def optimize_geometry(self, geometry, metal_name, ligand_name, m_data, l_data):
        R_base = m_data["radius"] + l_data["radius"]

        def objective(R_test):
            res = self.evaluate_energy_at_R(R_test, geometry, metal_name, ligand_name, m_data, l_data, R_base)
            return res["total_energy"]

        opt_res = opt.minimize_scalar(objective, bounds=(R_base * 0.7, R_base * 1.6), method='bounded')
        best_R = opt_res.x
        res = self.evaluate_energy_at_R(best_R, geometry, metal_name, ligand_name, m_data, l_data, R_base)

        res['stability_status'] = "STABLE"
        if best_R >= R_base * 1.45:
            res['stability_status'] = "REJECTED (Dissociated)"
            return res
        if res['total_energy'] > 0:
            res['stability_status'] = "REJECTED (Unbound)"
            return res

        return res

    def run_thermodynamic_race(self, metal_choice, ligand_choice):
        m_data = METAL_DATA[metal_choice]
        l_data = LIGAND_DATA[ligand_choice]
        results = {}
        for geo in ['Oh', 'Td', 'Sp']:
            results[geo] = self.optimize_geometry(geo, metal_choice, ligand_choice, m_data, l_data)
        return results

# ======================================================================
# STREAMLIT APP INTEGRATION
# ======================================================================
st.set_page_config(page_title="Coordination Complex Engine", layout="wide")

@st.cache_resource
def get_quantum_engine():
    return QuantumEngine()

engine = get_quantum_engine()
minimizer = AbInitioMinimizer(engine)

st.title("Semi-Emprical Coordination Complex Engine")
st.markdown("Matrix mechanics and crystal field theory-based semi-emprical geometry optimization tool.")

# Sidebar
st.sidebar.header("Selection of Metal and Ligand")
selected_metal = st.sidebar.selectbox("Select Metal", list(METAL_DATA.keys()))
selected_ligand = st.sidebar.selectbox("Select Ligand", list(LIGAND_DATA.keys()))

st.sidebar.markdown("---")
st.sidebar.markdown(
    "If you observe any wrong geometry, you can contact me at: "
    "[merthan.aytekin@metu.edu.tr](mailto:merthan.aytekin@metu.edu.tr)",
    unsafe_allow_html=True
)

if st.sidebar.button("Run Simulation", type="primary"):
    with st.spinner("Running quantum mechanical race across geometries..."):
        thermo_results = minimizer.run_thermodynamic_race(selected_metal, selected_ligand)
        valid_results = {k: v for k, v in thermo_results.items() if v['stability_status'] == "STABLE"}

    if not valid_results:
        st.error("System is thermodynamically UNSTABLE in all geometries (Dissociated or Positive Energy).")
    else:
        winner = min(valid_results, key=lambda k: valid_results[k]['total_energy'])
        win_data = thermo_results[winner]

        st.success(f"Global minimum found at **{winner}** geometry!")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Winning Geometry", winner)
        col2.metric("Min Binding Energy", f"{win_data['total_energy']:.2f} eV")
        col3.metric("Equilibrium Bond Length", f"{win_data['R_eq']:.3f} Å")
        col4.metric("Net Spin (S)", f"{win_data['spin']}")

        def get_symmetry_label(geo, orbital):
            if geo == 'Oh':
                return 'eg' if orbital in ['dz2', 'dx2-y2'] else 't2g'
            elif geo == 'Td':
                return 'e' if orbital in ['dz2', 'dx2-y2'] else 't2'
            elif geo == 'Sp':
                if orbital == 'dx2-y2': return 'b1g'
                elif orbital == 'dz2': return 'a1g'
                elif orbital == 'dxy': return 'b2g'
                else: return 'eg'
            return orbital

        tab1, tab2, tab3 = st.tabs(["Energy Plot", "Orbital Population", "Energy Levels Details"])

        with tab1:
            st.subheader("Orbital Splitting Diagram")
            fig, ax = plt.subplots(figsize=(8, 5))
            
            energies = [e for _, e in win_data['labeled_energies']]
            min_e, max_e = min(energies), max(energies)
            span = max_e - min_e if max_e != min_e else 1.0
            offset = span * 0.20
            
            for idx, (label, e) in enumerate(win_data['labeled_energies']):
                pop = win_data['config'][idx]
                color = 'royalblue' if pop > 0 else 'gray'
                sym_label = get_symmetry_label(winner, label)
                
                ax.hlines(e, idx - 0.35, idx + 0.35, colors=color, lw=4)
                ax.text(idx, e - offset * 0.5, f"{sym_label}\n({label})", ha='center', va='top', fontsize=10, fontweight='bold', color='black')
                
                if pop == 2:
                    elec_str = "↑↓"
                elif pop == 1:
                    elec_str = "↑"
                else:
                    elec_str = ""
                    
                if elec_str:
                    ax.text(idx, e + offset * 0.1, elec_str, ha='center', va='bottom', fontsize=14, fontweight='bold', color='darkred')
            
            ax.axhline(0, color='red', linestyle='--', linewidth=1.2, label='Barycenter')
            ax.set_xlim(-0.8, 4.8)
            ax.set_ylim(min_e - offset * 1.2, max_e + offset * 1.2)
            ax.set_xticks([])
            ax.set_ylabel("Energy relative to Barycenter (eV)", fontsize=12)
            ax.set_title(f"d-Orbital Energy Diagram for {selected_metal} + {selected_ligand} ({winner})", fontsize=13)
            ax.legend(loc='upper right')
            ax.grid(axis='y', linestyle=':', alpha=0.5)
            
            st.pyplot(fig)

        with tab2:
            st.subheader("Electronic Properties & Ground State")
            m_d_electrons = METAL_DATA[selected_metal]["d_electrons"]
            unpaired_count = int(win_data['spin'] * 2)
            
            st.write(f"- **System:** {selected_metal} ($d^{m_d_electrons}$)")
            st.write(f"- **Magnetism:** **{'PARAMAGNETIC' if win_data['spin'] > 0 else 'DIAMAGNETIC'}**")
            st.write(f"- **Unpaired Electrons:** {unpaired_count}")
            st.write(f"- **Pure LFSE:** {win_data['lfse']:+.4f} eV")
            
            st.markdown("#### Orbital Configurations")
            for idx, (label, e) in enumerate(win_data['labeled_energies']):
                pop = win_data['config'][idx]
                sym_label = get_symmetry_label(winner, label)
                
                if pop == 2:
                    pop_str = "[ ↑↓ ]"
                elif pop == 1:
                    pop_str = "[  ↑  ]"
                else:
                    pop_str = "[     ]"
                # eV bilgisi çıkarıldı, simetri etiketi ve orbital adı korundu
                st.code(f"{sym_label.upper():4s} ({label:7s}) : {pop_str}")

        with tab3:
            st.subheader("Dynamically Centered Energies")
            for label, e in win_data['labeled_energies']:
                sym_label = get_symmetry_label(winner, label)
                st.code(f"{sym_label.upper():4s} ({label:7s}) = {e:+.4f} eV")
            
            energies_only = [e[1] for e in win_data['labeled_energies']]
            delta_val = max(energies_only) - min(energies_only)
            st.write(f"**Splitting Energy ($\Delta$):** {delta_val:.4f} eV")
