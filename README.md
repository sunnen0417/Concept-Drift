# Concept-Drift
NTU MSLAB Concept drift team
* Hsu, Tsu-Yuan's branch

## electricity.py
Usage
1. simply copy the code into datasets.py
2. `from electricity import ElectricityDataset`
Note
1. some days may be dropped
	* e.g. batch\_days = 30 -> num\_batch = 943 // 30 (931-th day ~ 943-th day are dropped)
