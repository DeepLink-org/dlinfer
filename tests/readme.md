# How to add model for CI
To improve the efficiency of test case execution, we have downloaded the hf model files to a specific path in advance for easy use in test cases. 
The path where the model files are stored is defined in the tests/e2e/config.yaml file with parameter model_path(/data2/share_data in our local machine).
If you want to add your model for testing.
## First download your model locally
create a folder named in "my_model" under model_path(ie. mixtral_model_data), and download your model from hugging face in /model_path/my_model/
## Second modify config.yml
Add my_model/hf_model(ie. mixtral_model_data/Mixtral-8x7B-Instruct-v0.1) to config.yml pytorch_chat_model parameter.
Models run in tp1 in default. If the model need to run in multiple-cards, modify parameter tp_config in config.yml by adding "hf_model:tp_num"(ie.Mixtral-8x7B-Instruct-v0.1: 2)

# How to run test locally
## step1
Modify model_path and log_path in config.yml to your local path.
## step2
Set environment viarable
`export INFEREXT_TEST_DIR=/path/to/Inferext/tests`
## step3
Run following command
`
!/bin/bash
#run model in 1tp
cd /path/to/tests/e2e && pytest ./ -s -x --alluredir=allure-results --clean-alluredir
#run model in 2tp
python ./test_model_2tp.py
`