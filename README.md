# CIAO

This repository includes the implementation of the ICML 2024 paper titled [**Open Ad Hoc Teamwork with Cooperative Game Theory**](https://arxiv.org/abs/2402.15259):
- environments: Wolfpack, LBF;
- experimental setups: variant settings of agent-type sets for training and testing;
- algorithms: CIAO-S, CIAO-C and their variants, as well as the baseline algorithm GPL.

The demos of experiments are shown on https://sites.google.com/view/ciao2024.

## 1. Requirements Following [GPL](https://proceedings.mlr.press/v139/rahman21a.html) Settings

**(1) Set up a Python environment with Python 3.7.12.** 

**(2) Install required packages, execute the following command:** 

```setup
pip install -r requirements.txt
```

**(3) Install environments.**

For either LBF or Wolfpack, please ensure to install the environments using the following commands:
```setup
cd <Environment Name>/env
pip install -e .
```

**NOTE**: We recommend setting two different virtual environments for `Wolfpack` and `LBF`, respectively, to avoid the potential conflicts between these two experimental environments.

**(4) A modified version of OpenAI gym is required.**

To do the necessary modifications to `gym`, check the directory of the `gym` package using

```setup
pip show gym
```

Assuming that the package is installed in `<DIR>`, replace `<DIR>/gym/vector/async_vector_env.py` with the `async_vector_env.py` we have provided. This can be achieved using the following command: 

```setup
cp async_vector_env.py <DIR>/gym/vector/async_vector_env.py
```


## 2. Training

The training codes of experiments are contained in `Wolfpack` and `LBF`, respectively. Full description of the hyperparameters and the architecture used in this work is provided in the appendix of our paper.

For either environment, run the following commands to train CIAO-S, CIAO-C and GPL:

```setup
cd <Environment Name>/algorithm
```

### (1) Experiments in Section 5.1 and 5.2: Experimental results on identical agent-type sets for training and testing
#### Maximum of 5 agents during test
**GPL**
```python
python main_mrf.py --graph="complete" --weight_regularizer=0.0 --pair_range="free" --indiv_range="free" --note <NOTE> --save_dir=<SAVE_DIR> --num_players_test=5
```

**CIAO-C**
```python
python main_mrf.py --graph="complete" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="pos" --note <NOTE> --save_dir=<SAVE_DIR> --num_players_test=5
```

**CIAO-S**
```python
python main_mrf.py --graph="star" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="pos" --note <NOTE> --save_dir=<SAVE_DIR> --num_players_test=5
```

**CIAO-C-ZI**
```python
python main_mrf.py --graph="complete" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="zero" --note <NOTE> --save_dir=<SAVE_DIR> --num_players_test=5
```

**CIAO-C-NI**
```python
python main_mrf.py --graph="complete" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="neg" --note <NOTE> --save_dir=<SAVE_DIR> --num_players_test=5
```

**CIAO-C-FI**
```python
python main_mrf.py --graph="complete" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="free" --note <NOTE> --save_dir=<SAVE_DIR> --num_players_test=5
```

**CIAO-C-NP**
```python
python main_mrf.py --graph="complete" --weight_regularizer=0.5 --pair_range="neg" --indiv_range="pos" --note <NOTE> --save_dir=<SAVE_DIR> --num_players_test=5
```

**CIAO-S-ZI**
```python
python main_mrf.py --graph="star" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="zero" --note <NOTE> --save_dir=<SAVE_DIR> --num_players_test=5
```

**CIAO-S-NI**
```python
python main_mrf.py --graph="star" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="neg" --note <NOTE> --save_dir=<SAVE_DIR> --num_players_test=5
```

**CIAO-S-FI**
```python
python main_mrf.py --graph="star" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="free" --note <NOTE> --save_dir=<SAVE_DIR> --num_players_test=5
```

**CIAO-S-NP**
```python
python main_mrf.py --graph="star" --weight_regularizer=0.5 --pair_range="neg" --indiv_range="pos" --note <NOTE> --save_dir=<SAVE_DIR> --num_players_test=5
```

#### Maximum of 9 agents during test
The scripts of all algorithms are the same as above, but only with change of `--num_players_test=5` to `--num_players_test=9`.

### (2) Experiments in Section 5.3: Validating that solving GPL optimization problem is an approximation of Bellman operator in OSB-CAG (with identical agent-type sets in training and testing)
Since the scripts of CIAO-C and CIAO-S are the same as that for the maximum of 5 agents, we only show the scripts of CIAO-S-Va and CIAO-C-Va as follows:

**CIAO-C-Va**
```python
python main_mrf1.py --graph="complete" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="pos" --note <NOTE> --save_dir=<SAVE_DIR> --num_players_test=5 --update_manner="variant"
```

**CIAO-S-Va**
```python
python main_mrf1.py --graph="star" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="pos" --note <NOTE> --save_dir=<SAVE_DIR> --num_players_test=5 --update_manner="variant"
```

### (3) Experiments in Appendix J.2 and J.3: Agent-type sets excluding A2C agent (still identical in training and testing and only for LBF)
#### Maximum of 5 agents during test
**GPL**
```python
python main_mrf.py --graph="complete" --weight_regularizer=0.0 --pair_range="free" --indiv_range="free" --note <NOTE> --save_dir=<SAVE_DIR> --num_players_test=5 --exclude_A2Cagent
```

**CIAO-C**
```python
python main_mrf.py --graph="complete" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="pos" --note <NOTE> --save_dir=<SAVE_DIR> --num_players_test=5 --exclude_A2Cagent
```

**CIAO-S**
```python
python main_mrf.py --graph="star" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="pos" --note <NOTE> --save_dir=<SAVE_DIR> --num_players_test=5 --exclude_A2Cagent
```

**CIAO-C-ZI**
```python
python main_mrf.py --graph="complete" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="zero" --note <NOTE> --save_dir=<SAVE_DIR> --num_players_test=5 --exclude_A2Cagent
```

**CIAO-C-NI**
```python
python main_mrf.py --graph="complete" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="neg" --note <NOTE> --save_dir=<SAVE_DIR> --num_players_test=5 --exclude_A2Cagent
```

**CIAO-C-FI**
```python
python main_mrf.py --graph="complete" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="free" --note <NOTE> --save_dir=<SAVE_DIR> --num_players_test=5 --exclude_A2Cagent
```

**CIAO-C-NP**
```python
python main_mrf.py --graph="complete" --weight_regularizer=0.5 --pair_range="neg" --indiv_range="pos" --note <NOTE> --save_dir=<SAVE_DIR> --num_players_test=5 --exclude_A2Cagent
```

**CIAO-S-ZI**
```python
python main_mrf.py --graph="star" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="zero" --note <NOTE> --save_dir=<SAVE_DIR> --num_players_test=5 --exclude_A2Cagent
```

**CIAO-S-NI**
```python
python main_mrf.py --graph="star" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="neg" --note <NOTE> --save_dir=<SAVE_DIR> --num_players_test=5 --exclude_A2Cagent
```

**CIAO-S-FI**
```python
python main_mrf.py --graph="star" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="free" --note <NOTE> --save_dir=<SAVE_DIR> --num_players_test=5 --exclude_A2Cagent
```

**CIAO-S-NP**
```python
python main_mrf.py --graph="star" --weight_regularizer=0.5 --pair_range="neg" --indiv_range="pos" --note <NOTE> --save_dir=<SAVE_DIR> --num_players_test=5 --exclude_A2Cagent
```

#### Maximum of 9 agents during test
The scripts of all algorithms are the same as above, but only with change of `--num_players_test=5` to `--num_players_test=9`.

### (4) Experiments in Appendix J.4: Generalizability of CIAO with different agent-type sets in training and testing

#### Agent-type sets in training and testing have intersection, with one shared agent-type
**GPL**
 ```python
 python main_mrf1.py --graph="complete" --weight_regularizer=0.0 --pair_range="free" --indiv_range="free" --note <NOTE> --save_dir=<SAVE_DIR> --num_players_test=5 --intersection_generalization
 ```

 **CIAO-C**
 ```python
 python main_mrf1.py --graph="complete" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="pos" --note <NOTE> --save_dir=<SAVE_DIR> --num_players_test=5 --intersection_generalization
 ```

 **CIAO-S**
 ```python
 python main_mrf1.py --graph="star" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="pos" --note <NOTE> --save_dir=<SAVE_DIR> --num_players_test=5 --intersection_generalization
 ```

#### Agent-type sets in training and testing are mutually exclusive
**GPL**
```python
python main_mrf1.py --graph="complete" --weight_regularizer=0.0 --pair_range="free" --indiv_range="free" --note <NOTE> --save_dir=<SAVE_DIR> --num_players_test=5 --exclusion_generalization
```

**CIAO-C**
```python
python main_mrf1.py --graph="complete" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="pos" --note <NOTE> --save_dir=<SAVE_DIR> --num_players_test=5 --exclusion_generalization
```

**CIAO-S**
```python
python main_mrf1.py --graph="star" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="pos" --note <NOTE> --save_dir=<SAVE_DIR> --num_players_test=5 --exclusion_generalization
```

### (5) Experiments in Appendix J.5: CIAO with no regularizers (agent-type sets being identical in training and testing)
Since the scripts of CIAO-C and CIAO-S are the same as that for experiments in Section 5.1 and 5.2 above, we only show the scripts of CIAO-S-NR and CIAO-C-NR as follows:

#### Maximum of 5 agents during test (LBF including A2C agent / Wolfpack)
**CIAO-C-NR**
```python
python main_mrf.py --graph="complete" --weight_regularizer=0.0 --pair_range="pos" --indiv_range="pos" --note <NOTE> --save_dir=<SAVE_DIR> --num_players_test=5
```

**CIAO-S-NR**
```python
python main_mrf.py --graph="star" --weight_regularizer=0.0 --pair_range="pos" --indiv_range="pos" --note <NOTE> --save_dir=<SAVE_DIR> --num_players_test=5
```

#### Maximum of 9 agents during test (LBF including A2C agent / Wolfpack)
**CIAO-C-NR**
```python
python main_mrf.py --graph="complete" --weight_regularizer=0.0 --pair_range="pos" --indiv_range="pos" --note <NOTE> --save_dir=<SAVE_DIR> --num_players_test=9
```

**CIAO-S-NR**
```python
python main_mrf.py --graph="star" --weight_regularizer=0.0 --pair_range="pos" --indiv_range="pos" --note <NOTE> --save_dir=<SAVE_DIR> --num_players_test=9
```

#### Maximum of 5 agents during test (LBF excluding A2C agent)
**CIAO-C-NR**
```python
python main_mrf.py --graph="complete" --weight_regularizer=0.0 --pair_range="pos" --indiv_range="pos" --note <NOTE> --save_dir=<SAVE_DIR> --num_players_test=5 --exclude_A2Cagent
```

**CIAO-S-NR**
```python
python main_mrf.py --graph="star" --weight_regularizer=0.0 --pair_range="pos" --indiv_range="pos" --note <NOTE> --save_dir=<SAVE_DIR> --num_players_test=5 --exclude_A2Cagent
```

#### Maximum of 9 agents during test (LBF excluding A2C agent)
**CIAO-C-NR**
```python
python main_mrf.py --graph="complete" --weight_regularizer=0.0 --pair_range="pos" --indiv_range="pos" --note <NOTE> --save_dir=<SAVE_DIR> --num_players_test=9 --exclude_A2Cagent
```

**CIAO-S-NR**
```python
python main_mrf.py --graph="star" --weight_regularizer=0.0 --pair_range="pos" --indiv_range="pos" --note <NOTE> --save_dir=<SAVE_DIR> --num_players_test=9 --exclude_A2Cagent
```

## 3. Monitoring Experimental Results
Aside from training models, the shell script also periodically checkpoints the model and evaluates it in the training and evaluation environment. We specifically run several episodes under the evaluation setup and log the resulting performance using tensorboard. The resulting logs can be viewed using the following command : 

```script
tensorboard --logdir=<Environment Name>/algorithm/runs
```

## Citing
If you would like to use the result of this paper, please cite the following paper:
```
@article{wang2024open,
  title={Open Ad Hoc Teamwork with Cooperative Game Theory},
  author={Wang, Jianhong and Li, Yang and Zhang, Yuan and Pan, Wei and Kaski, Samuel},
  journal={arXiv preprint arXiv:2402.15259},
  year={2024}
}
```
## Contact
If you have any queries about this paper, please drop an email to jianhong.wang@manchester.ac.uk.