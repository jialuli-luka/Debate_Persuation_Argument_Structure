# Exploring the Role of Argument Structure in Online Debate Persuasion

Code for paper "Exploring the Role of Argument Structure in Online Debate Persuasion" (https://www.aclweb.org/anthology/2020.emnlp-main.716.pdf) in EMNLP 2020.  
Data preparation:
Download DDO data from https://www.cs.cornell.edu/~esindurmus/ddo.html

Generate Argument Structure follow https://github.com/vene/marseille (use SVM  & strict option)
save the generated argument structure for each debate under src/data/. (each debate saved as one json file)

Filter data:
python src/generation_stat.py

To replicate the full model in the paper:
1. Filter out data based on criterion (e.g. length, vote difference)
(Change CORENLP_PATH at the beginning of the file)
python src/BERT/data_generation.py

2. Generate BERT representation for debates
python src/BERT/BERT_representation_generation.py

3. Train full model:
python src/BERT/RNN_with_BERT_experiment.py full

4. Train model w/o Argument Structure:
python src/BERT/RNN_with_BERT_experiment.py linguistic


To replicate LR baselines in the paper:
python src/experiment.py
