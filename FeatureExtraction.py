"""
Extract the following features for classification:
MFCCs with deltas and delta deltas
Energy
Fundamental frequency?
Pitch?
"""

import speechpy as sp
import InputPreparation

print(sp.processing.preemphasis(InputPreparation.convert_audio("Data/")[1]))
