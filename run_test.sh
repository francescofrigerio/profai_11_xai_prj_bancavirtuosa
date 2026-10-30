#!/bin/bash

# Dir di lavoro
cd tests || exit
ls training_test.py
# Esecuzione del codice Python

# 

# python -m pytest -v -s -k "test_load_checkpoint_field" 2>&1 | tee run_test_1.log

python -m pytest -k "test_load_checkpoint_weights_only" | tee run_test_2.log

# python -m pytest -k "test_model_evaluation" | tee run_test_3.log


