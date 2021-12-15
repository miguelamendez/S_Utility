This repo contains all the libraries and code necessary to replicate the experiments conducted in the paper : Semantic Utility.

Installlation guide:
The code requires the follow dependencies to be installed.

Python basic Libraries:
pip install pickle
pip install numpy 
pip install matplotlib

DeepLearning:
PyTorch:https://pytorch.org/
pip install torch

Environments:
GYM: https://github.com/openai/gym
pip install gym
ALE: https://github.com/mgbellemare/Arcade-Learning-Environment
pip install ale-py
For Atari roms make sure to download them (http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html) and run:
ale-import-roms /path/to/roms/

Symbolic AI:
CNFgen:https://massimolauria.net/cnfgen/
pip3 install [--user] cnfgen

WMC/SDD library:https://pysdd.readthedocs.io/en/latest/
pip install pysdd

Sat solvers:https://github.com/pysathq/pysat
pip install python-sat[pblib,aiger]

Others:
Graph Analysis :https://networkx.org/documentation/stable/install.html
pip install networkx[default]
