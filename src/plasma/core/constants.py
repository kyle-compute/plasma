"""SI physical constants used throughout the simulation."""

import math

# Fundamental constants (CODATA 2018)
E_CHARGE = 1.602176634e-19       # Elementary charge [C]
M_ELECTRON = 9.1093837015e-31    # Electron mass [kg]
M_PROTON = 1.67262192369e-27     # Proton mass [kg]
EPSILON_0 = 8.8541878128e-12     # Vacuum permittivity [F/m]
MU_0 = 1.25663706212e-6          # Vacuum permeability [H/m]
K_BOLTZMANN = 1.380649e-23       # Boltzmann constant [J/K]
AMU = 1.66053906660e-27          # Atomic mass unit [kg]
PI = math.pi
TWO_PI = 2.0 * PI

# Derived constants
EV_TO_JOULE = E_CHARGE           # 1 eV = 1.602e-19 J
JOULE_TO_EV = 1.0 / EV_TO_JOULE
EV_TO_KELVIN = E_CHARGE / K_BOLTZMANN  # 1 eV ≈ 11604.5 K

# Species masses [kg]
M_AR = 39.948 * AMU              # Argon
M_CU = 63.546 * AMU              # Copper
M_TI = 47.867 * AMU              # Titanium
M_AL = 26.982 * AMU              # Aluminum
MATERIAL_MASSES = {
    "Ar": M_AR,
    "Cu": M_CU,
    "Ti": M_TI,
    "Al": M_AL,
}

# Common thresholds [eV]
E_IONIZATION_AR = 15.76          # Ar ground state ionization
E_METASTABLE_AR = 11.55          # Ar(4s[3/2]_2) metastable
E_RESONANT_AR = 11.72            # Ar(4s'[1/2]_0) resonant
E_IONIZATION_CU = 7.726          # Cu ground state ionization
E_IONIZATION_TI = 6.828          # Ti ground state ionization


def material_mass_kg(symbol: str) -> float:
    """Return the mass of one named material species in kilograms."""

    try:
        return MATERIAL_MASSES[symbol]
    except KeyError as exc:
        raise KeyError(f"Unsupported material '{symbol}'.") from exc
