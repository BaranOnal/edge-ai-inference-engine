Data Set: FD001
Train trjectories: 100
Test trajectories: 100
Conditions: ONE (Sea Level)
Fault Modes: ONE (HPC Degradation)

Data Set: FD002
Train trjectories: 260
Test trajectories: 259
Conditions: SIX 
Fault Modes: ONE (HPC Degradation)

Data Set: FD003
Train trjectories: 100
Test trajectories: 100
Conditions: ONE (Sea Level)
Fault Modes: TWO (HPC Degradation, Fan Degradation)

Data Set: FD004
Train trjectories: 248
Test trajectories: 249
Conditions: SIX 
Fault Modes: TWO (HPC Degradation, Fan Degradation)



Experimental Scenario

Data sets consists of multiple multivariate time series. Each data set is further divided into training and test subsets. Each time series is from a different engine � i.e., the data can be considered to be from a fleet of engines of the same type. Each engine starts with different degrees of initial wear and manufacturing variation which is unknown to the user. This wear and variation is considered normal, i.e., it is not considered a fault condition. There are three operational settings that have a substantial effect on engine performance. These settings are also included in the data. The data is contaminated with sensor noise.

The engine is operating normally at the start of each time series, and develops a fault at some point during the series. In the training set, the fault grows in magnitude until system failure. In the test set, the time series ends some time prior to system failure. The objective of the competition is to predict the number of remaining operational cycles before failure in the test set, i.e., the number of operational cycles after the last cycle that the engine will continue to operate. Also provided a vector of true Remaining Useful Life (RUL) values for the test data.

The data are provided as a zip-compressed text file with 26 columns of numbers, separated by spaces. Each row is a snapshot of data taken during a single operational cycle, each column is a different variable. The columns correspond to:
1)	unit number
2)	time, in cycles
3)	operational setting 1
4)	operational setting 2
5)	operational setting 3
6)	sensor measurement  1
7)	sensor measurement  2
...
26)	sensor measurement  26


Reference: A. Saxena, K. Goebel, D. Simon, and N. Eklund, �Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation�, in the Proceedings of the Ist International Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.


# ==============================================================================
# NASA CMAPSS TURBOFAN ENGINE SENSOR DESCRIPTIONS
# ------------------------------------------------------------------------------
# Indices below correspond to the 21 sensor columns (s1 to s21) in the dataset.
# Note: In train_FD001.txt, operating conditions are constant, so sensors 
# measuring ambient or demanded values (like s1, s5, s10, s16, s18, s19) 
# will have zero variance and should be excluded from training.
# ==============================================================================

# s1:  T24       - Total temperature at fan inlet (°R)
# s2:  T30       - Total temperature at LPC (Low Pressure Compressor) outlet (°R)
# s3:  T50       - Total temperature at HPC (High Pressure Compressor) outlet (°R)
# s4:  T52       - Total temperature at LPT (Low Pressure Turbine) outlet (°R)
# s5:  P2        - Pressure at fan inlet (psia)
# s6:  P15       - Total pressure in bypass-duct (psia)
# s7:  P30       - Total pressure at HPC outlet (psia)
# s8:  Nf        - Physical fan speed (rpm)
# s9:  Nc        - Physical core speed (rpm)
# s10: epr       - Engine pressure ratio (P50/P2)
# s11: Ps30      - Static pressure at HPC outlet (psia)
# s12: phi       - Ratio of fuel flow to Ps30 (pps/psi)
# s13: NRf       - Corrected fan speed (rpm)
# s14: NRc       - Corrected core speed (rpm)
# s15: BPR       - Bypass Ratio (-)
# s16: farB      - Burner fuel-air ratio (-)
# s17: htBleed   - Bleed Enthalpy (-)
# s18: Nf_dmd    - Demanded fan speed (rpm)
# s19: PCNfR_dmd - Demanded corrected fan speed (rpm)
# s20: W31       - HPT (High Pressure Turbine) coolant bleed (lbm/s)
# s21: W32       - LPT (Low Pressure Turbine) coolant bleed (lbm/s)

# Selected features for Edge AI prediction (highest correlation with RUL):
# SENSORS = ['s2', 's3', 's4', 's7']