#!/bin/bash

# Dir di lavoro
cd tests|| exit

# Esecuzione del codice Python
# python -m pytest -k "test_load_checkpoint" | tee run_test_1.log

python -m pytest -k "test_model_evaluation" | tee run_test_2.log


