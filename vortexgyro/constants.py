"""Physical constants and normalized gyro-Bohm reference scales."""

from dataclasses import dataclass
import numpy as np

# Physical constants in SI
E_CHARGE = 1.602176634e-19       # Elementary charge [C]
M_PROTON = 1.67262192369e-27     # Proton mass [kg]
M_ELECTRON = 9.1093837015e-31    # Electron mass [kg]
KB = 1.380649e-23                # Boltzmann constant [J/K]
EPS0 = 8.8541878128e-12          # Vacuum permittivity [F/m]


@dataclass
class PlasmaReference:
    """Tokamak plasma reference parameters for gyro-Bohm normalization."""
    B0: float = 2.5               # Magnetic guide field [Tesla]
    Te_eV: float = 2000.0         # Electron temperature [eV]
    Ti_eV: float = 2000.0         # Ion temperature [eV]
    n0: float = 5.0e19            # Core electron density [m^-3]
    R0: float = 1.65              # Major radius [m]
    a: float = 0.50               # Minor radius [m]
    ion_mass_amu: float = 2.0     # Deuterium plasma (2.0 AMU)

    @property
    def ion_mass(self) -> float:
        return self.ion_mass_amu * M_PROTON

    @property
    def sound_speed(self) -> float:
        """Sound speed c_s = sqrt(T_e / m_i) [m/s]."""
        te_joules = self.Te_eV * E_CHARGE
        return np.sqrt(te_joules / self.ion_mass)

    @property
    def ion_cyclotron_freq(self) -> float:
        """Ion gyrofrequency Omega_i = e * B / m_i [rad/s]."""
        return (E_CHARGE * self.B0) / self.ion_mass

    @property
    def sound_gyroradius(self) -> float:
        """Sound Larmor radius rho_s = c_s / Omega_i [m]."""
        return self.sound_speed / self.ion_cyclotron_freq

    @property
    def rho_star(self) -> float:
        """Normalized gyroradius rho* = rho_s / a."""
        return self.sound_gyroradius / self.a

    @property
    def tau(self) -> float:
        """Temperature ratio tau = T_i / T_e."""
        return self.Ti_eV / self.Te_eV
