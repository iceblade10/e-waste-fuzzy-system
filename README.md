# e-waste-fuzzy-system
Academic Context
Course: CSC3034 - Computational Intelligence
Assignment 1: Waste-Sorting Assistance for Electronic Waste
Developed by: [Kiroshan Ram] [Myles Lim Wenn Liang] [Ngoi Yi Ming] [Chow De Xian] [William Foong Mun Kit] - Fuzzy System Design & Implementation

A Sugeno-type fuzzy inference system (FIS) designed to support electronic waste (e-waste) sorting and recycling decisions.  
This system evaluates four key input factors which are metal content, contamination level, repairability, and hazard rating to classify e-waste items into safe and sustainable handling routes.
System Overview

| Input Variable | Range | Membership Type | Linguistic Terms |
| Metal Content (%) | 0–100 | Trapezoidal / Triangular | Low, Medium, High |
| Contamination (0–100) | 0–100 | Triangular | Low, Moderate, High |
| Repairability (0–10) | 0–10 | Triangular | Poor, Moderate, Good |
| Hazard (0–10) | 0–10 | Trapezoidal / Triangular | Safe, Risky, Hazardous |

Output:
A decision score (0–100) mapped to handling categories:

| Handling Category | Output Constant |
| Refurbish / Reuse | 25 |
| Material Recovery | 50 |
| Hazardous Handling | 75 |
| Safe Disposal | 90 |

The rule base prioritizes safety and environmental protection, while also encouraging recoverability of materials.

Usage
Run demo cases
To test the system with built-in sample inputs:
python e_waste_fis.py –demo

To save visualizations (MFs and decision surfaces):
python e_waste_fis.py --figures --out results

To import and evaluate new cases directly:
from e_waste_fis import sugeno_infer
print(sugeno_infer(85, 10, 3, 1))

Running --figures produces:
•	mf_metal.png
•	mf_contamination.png
•	mf_repairability.png
•	mf_hazard.png
•	surface_metal_vs_contamination.png
•	surface_repairability_vs_hazard.png
•	slice_contamination.png
\
These illustrate how the fuzzy logic surface evolves across input dimensions.

Dependencies
•	Python ≥ 3.10
•	NumPy
•	Matplotlib

