# Heat_Mass_Transfer
This repository is a library of functions used in heat and mass transfer.

## Quick Start
First load the environment by heading to the heat_mass_transfer directory, entering conda env -f environment.yml and following its instructions to load the environment.

In the command-line terminal type in python heat_transfer_gui.py. Select the type of heat transfer occurring. Then enter the known values to then calculate for the unknown value.

Alternatively, if you have a system that needs solving, type in python simulation.py and follow the instructions to best describe your system.

## Source Files
# conduction.py
Implements basic heat transfer through conduction functions, including 1D conduction, conduction through cylinders, and conduction through spheres.

# free_convection.py
Implements free convection calculations and its associated dimensionless numbers. Often used in ambient or quiescent conditions.

# forced_convection.py
Implements forced convection, including laminar and turbulent regimes. Their associated dimensionless numbers. Often used in instances where there's high winds, fans, or fast fluid flows.

# radiation.py
Implements radiative heat transfer, emissivity, gray bodies, and black bodies.

# transient.py
Implements time-dependent (transient) heat transfer methods, almost exclusively conduction. For example, think of searing a tear for a short period of time.

### DISCLAIMER!
AI was used to help create heat_transfer_gui.py.