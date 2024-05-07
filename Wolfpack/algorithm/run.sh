#!/bin/bash

export OMP_NUM_THREADS=1

# 5 test agents
# GPL
# python main_mrf1.py --graph="complete" --weight_regularizer=0.0 --pair_range="free" --indiv_range="free" --note gpl-$SLURM_ARRAY_TASK_ID --save_dir="/scratch/project_2008459/jianhong/adhoc-teamwork/new/Wolfpack/test-5" --num_players_test=5
# CIAOC
# python main_mrf1.py --graph="complete" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="pos" --note ciaoc-$SLURM_ARRAY_TASK_ID --save_dir="/scratch/project_2008459/jianhong/adhoc-teamwork/new/Wolfpack/test-5" --num_players_test=5
# CIAOS
# python main_mrf1.py --graph="star" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="pos" --note ciaos-$SLURM_ARRAY_TASK_ID --save_dir="/scratch/project_2008459/jianhong/adhoc-teamwork/new/Wolfpack/test-5" --num_players_test=5
# CIAOC-ZERO_INDIV
# python main_mrf1.py --graph="complete" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="zero" --note ciaoc-zi-$SLURM_ARRAY_TASK_ID --save_dir="/scratch/project_2008459/jianhong/adhoc-teamwork/new/Wolfpack/test-5" --num_players_test=5
# CIAOC-NEG_INDIV
# python main_mrf1.py --graph="complete" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="neg" --note ciaoc-ni-$SLURM_ARRAY_TASK_ID --save_dir="/scratch/project_2008459/jianhong/adhoc-teamwork/new/Wolfpack/test-5" --num_players_test=5
# CIAOC-FREE_INDIV
# python main_mrf1.py --graph="complete" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="free" --note ciaoc-fi-$SLURM_ARRAY_TASK_ID --save_dir="/scratch/project_2008459/jianhong/adhoc-teamwork/new/Wolfpack/test-5" --num_players_test=5
# CIAOC-NEG_PAIR
# python main_mrf1.py --graph="complete" --weight_regularizer=0.5 --pair_range="neg" --indiv_range="pos" --note ciaoc-np-$SLURM_ARRAY_TASK_ID --save_dir="/scratch/project_2008459/jianhong/adhoc-teamwork/new/Wolfpack/test-5" --num_players_test=5
# CIAOS-ZERO_INDIV
# python main_mrf1.py --graph="star" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="zero" --note ciaos-zi-$SLURM_ARRAY_TASK_ID --save_dir="/scratch/project_2008459/jianhong/adhoc-teamwork/new/Wolfpack/test-5" --num_players_test=5
# CIAOS-NEG_INDIV
# python main_mrf1.py --graph="star" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="neg" --note ciaos-ni-$SLURM_ARRAY_TASK_ID --save_dir="/scratch/project_2008459/jianhong/adhoc-teamwork/new/Wolfpack/test-5" --num_players_test=5
# CIAOS-FREE_INDIV
# python main_mrf1.py --graph="star" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="free" --note ciaos-fi-$SLURM_ARRAY_TASK_ID --save_dir="/scratch/project_2008459/jianhong/adhoc-teamwork/new/Wolfpack/test-5" --num_players_test=5
# CIAOS-NEG_PAIR
# python main_mrf1.py --graph="star" --weight_regularizer=0.5 --pair_range="neg" --indiv_range="pos" --note ciaos-np-$SLURM_ARRAY_TASK_ID --save_dir="/scratch/project_2008459/jianhong/adhoc-teamwork/new/Wolfpack/test-5" --num_players_test=5
# CIAOC-NO_REG
# python main_mrf1.py --graph="complete" --weight_regularizer=0.0 --pair_range="pos" --indiv_range="pos" --note ciaoc-noReg-$SLURM_ARRAY_TASK_ID --save_dir="/scratch/project_2008459/jianhong/adhoc-teamwork/new/Wolfpack/test-5" --num_players_test=5
# CIAOS-NO_REG
# python main_mrf1.py --graph="star" --weight_regularizer=0.0 --pair_range="pos" --indiv_range="pos" --note ciaos-noReg-$SLURM_ARRAY_TASK_ID --save_dir="/scratch/project_2008459/jianhong/adhoc-teamwork/new/Wolfpack/test-5" --num_players_test=5
# CIAOC with variant update
# python main_mrf1.py --graph="complete" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="pos" --note ciaoc-variant-$SLURM_ARRAY_TASK_ID --save_dir="/scratch/project_2008459/jianhong/adhoc-teamwork/new/Wolfpack/test-5" --num_players_test=5 --update_manner variant
# CIAOS with variant update
# python main_mrf1.py --graph="star" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="pos" --note ciaos-variant-$SLURM_ARRAY_TASK_ID --save_dir="/scratch/project_2008459/jianhong/adhoc-teamwork/new/Wolfpack/test-5" --num_players_test=5 --update_manner variant

# 9 test agents
# GPL
# python main_mrf1.py --graph="complete" --weight_regularizer=0.0 --pair_range="free" --indiv_range="free" --note gpl-$SLURM_ARRAY_TASK_ID --save_dir="/scratch/project_2008459/jianhong/adhoc-teamwork/new/Wolfpack/test-9" --num_players_test=9
# CIAOC
# python main_mrf1.py --graph="complete" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="pos" --note ciaoc-$SLURM_ARRAY_TASK_ID --save_dir="/scratch/project_2008459/jianhong/adhoc-teamwork/new/Wolfpack/test-9" --num_players_test=9
# CIAOS
# python main_mrf1.py --graph="star" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="pos" --note ciaos-$SLURM_ARRAY_TASK_ID --save_dir="/scratch/project_2008459/jianhong/adhoc-teamwork/new/Wolfpack/test-9" --num_players_test=9
# CIAOC-ZERO_INDIV
# python main_mrf1.py --graph="complete" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="zero" --note ciaoc-zi-$SLURM_ARRAY_TASK_ID --save_dir="/scratch/project_2008459/jianhong/adhoc-teamwork/new/Wolfpack/test-9" --num_players_test=9
# CIAOC-NEG_INDIV
# python main_mrf1.py --graph="complete" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="neg" --note ciaoc-ni-$SLURM_ARRAY_TASK_ID --save_dir="/scratch/project_2008459/jianhong/adhoc-teamwork/new/Wolfpack/test-9" --num_players_test=9
# CIAOC-FREE_INDIV
# python main_mrf1.py --graph="complete" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="free" --note ciaoc-fi-$SLURM_ARRAY_TASK_ID --save_dir="/scratch/project_2008459/jianhong/adhoc-teamwork/new/Wolfpack/test-9" --num_players_test=9
# CIAOC-NEG_PAIR
# python main_mrf1.py --graph="complete" --weight_regularizer=0.5 --pair_range="neg" --indiv_range="pos" --note ciaoc-np-$SLURM_ARRAY_TASK_ID --save_dir="/scratch/project_2008459/jianhong/adhoc-teamwork/new/Wolfpack/test-9" --num_players_test=9
# CIAOS-ZERO_INDIV
# python main_mrf1.py --graph="star" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="zero" --note ciaos-zi-$SLURM_ARRAY_TASK_ID --save_dir="/scratch/project_2008459/jianhong/adhoc-teamwork/new/Wolfpack/test-9" --num_players_test=9
# CIAOS-NEG_INDIV
# python main_mrf1.py --graph="star" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="neg" --note ciaos-ni-$SLURM_ARRAY_TASK_ID --save_dir="/scratch/project_2008459/jianhong/adhoc-teamwork/new/Wolfpack/test-9" --num_players_test=9
# CIAOS-FREE_INDIV
# python main_mrf1.py --graph="star" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="free" --note ciaos-fi-$SLURM_ARRAY_TASK_ID --save_dir="/scratch/project_2008459/jianhong/adhoc-teamwork/new/Wolfpack/test-9" --num_players_test=9
# CIAOS-NEG_PAIR
# python main_mrf1.py --graph="star" --weight_regularizer=0.5 --pair_range="neg" --indiv_range="pos" --note ciaos-np-$SLURM_ARRAY_TASK_ID --save_dir="/scratch/project_2008459/jianhong/adhoc-teamwork/new/Wolfpack/test-9" --num_players_test=9
# CIAOC-NO_REG
# python main_mrf1.py --graph="complete" --weight_regularizer=0.0 --pair_range="pos" --indiv_range="pos" --note ciaoc-noReg-$SLURM_ARRAY_TASK_ID --save_dir="/scratch/project_2008459/jianhong/adhoc-teamwork/new/Wolfpack/test-9" --num_players_test=9
# CIAOS-NO_REG
# python main_mrf1.py --graph="star" --weight_regularizer=0.0 --pair_range="pos" --indiv_range="pos" --note ciaos-noReg-$SLURM_ARRAY_TASK_ID --save_dir="/scratch/project_2008459/jianhong/adhoc-teamwork/new/Wolfpack/test-9" --num_players_test=9
# CIAOC with variant update
# python main_mrf1.py --graph="complete" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="pos" --note ciaoc-variant-$SLURM_ARRAY_TASK_ID --save_dir="/scratch/project_2008459/jianhong/adhoc-teamwork/new/Wolfpack/test-9" --num_players_test=9 --update_manner variant
# CIAOS with variant update
# python main_mrf1.py --graph="star" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="pos" --note ciaos-variant-$SLURM_ARRAY_TASK_ID --save_dir="/scratch/project_2008459/jianhong/adhoc-teamwork/new/Wolfpack/test-9" --num_players_test=9 --update_manner variant

# intersection generalization
# GPL
# python main_mrf1.py --graph="complete" --weight_regularizer=0.0 --pair_range="free" --indiv_range="free" --note gpl-intersection-$SLURM_ARRAY_TASK_ID --save_dir="/scratch/project_2008459/jianhong/adhoc-teamwork/new/Wolfpack/test-5" --num_players_test=5 --intersection_generalization
# CIAOC
# python main_mrf1.py --graph="complete" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="pos" --note ciaoc-intersection-$SLURM_ARRAY_TASK_ID --save_dir="/scratch/project_2008459/jianhong/adhoc-teamwork/new/Wolfpack/test-5" --num_players_test=5 --intersection_generalization
# CIAOS
# python main_mrf1.py --graph="star" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="pos" --note ciaos-intersection-$SLURM_ARRAY_TASK_ID --save_dir="/scratch/project_2008459/jianhong/adhoc-teamwork/new/Wolfpack/test-5" --num_players_test=5 --intersection_generalization

# exclusion generalization
# GPL
# python main_mrf1.py --graph="complete" --weight_regularizer=0.0 --pair_range="free" --indiv_range="free" --note gpl-exclusion-$SLURM_ARRAY_TASK_ID --save_dir="/scratch/project_2008459/jianhong/adhoc-teamwork/new/Wolfpack/test-5" --num_players_test=5 --exclusion_generalization
# CIAOC
# python main_mrf1.py --graph="complete" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="pos" --note ciaoc-exclusion-$SLURM_ARRAY_TASK_ID --save_dir="/scratch/project_2008459/jianhong/adhoc-teamwork/new/Wolfpack/test-5" --num_players_test=5 --exclusion_generalization
# CIAOS
# python main_mrf1.py --graph="star" --weight_regularizer=0.5 --pair_range="pos" --indiv_range="pos" --note ciaos-exclusion-$SLURM_ARRAY_TASK_ID --save_dir="/scratch/project_2008459/jianhong/adhoc-teamwork/new/Wolfpack/test-5" --num_players_test=5 --exclusion_generalization