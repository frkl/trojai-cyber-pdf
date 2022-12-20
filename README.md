This repo contains a the code for TrinitySRITrojAI submission to the cyber-pdf-dec2022 round of the [TrojAI leaderboard](https://pages.nist.gov/trojai/). 

# Dependencies

First install the dependencies required by https://github.com/usnistgov/trojai-example.

Additional dependencies
```
pip install pandas
pip install scipy
pip install scikit-learn
```

# Usage

Clone code into `<root>/trojai-cyber-pdf`

Download and extract training set `cyber-pdf-dec2022-train.tar.gz` into `<root>/trojai-datasets/cyber-pdf-dec2022-train`

`cd` into `<root>/trojai-cyber-pdf` and run the following commands

First run feature extraction on the training set.

```
python weight_analysis.py
```

This will produce a feature file `<root>/trojai-cyber-pdf/data_cyber-pdf_weight.pt`.

Then run cross-validation hyperparameter search using the feature file and a pre-defined detector architecture

```
python crossval_folds.py --arch arch.mlp_set5 --data data_cyber-pdf_weight.pt --nsplits 7
```

This will produce a set of learned detector parameters at `<root>/trojai-cyber-pdf/sessions/000000/`. 

Finally copy the detector parameters into a `learned_parameters` folder and build the singularity container.
```
cp -r ./sessions/000000/ ./learned_parameters
./build.sh
```

The script will test inference functionalities and build a container at `cyber-pdf-dec2022_sts_SRI_weight_grad_v1.simg`.


