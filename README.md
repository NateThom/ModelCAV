# ModelCAV

## Setup
Install dependencies:
```console
conda env create -f environment.yml
```

Clone opensphere codebase. \
With HTTPS:
```console
git clone https://github.com/NateThom/opensphere.git
```
With SSH:
```console
git clone git@github.com:NateThom/opensphere.git
```

Download datasets for opensphere:
```console
cd opensphere
bash scripts/dataset_setup.sh
```