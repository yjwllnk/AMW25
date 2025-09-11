!! Description
This repository containts a Python package of Team 4(Dr. Abhijeet, Chaehyun-Moon, Yujin/Will Kang) of the 2025 KRICT ChemDX Hackathon

Our goal is to explore  the effect of symmetry operations on the thermoelectric properties of material, and this package's primary purpose is to try out different ML models & techniques for data analysis


!! Installation
```
    git clone git@github.com:boffintocoffin/AMW25.git
    cd ./AMW25
    pip install .
```

!! CLI Usage

```
    amw25 --cwd <path_to_working_directory> --config <configuration_file> --model [linear_regressor, xgb, mlp]
```

!! env 
```
    micromamba create -n amw python==3.11
    micromamba activate amw
```

