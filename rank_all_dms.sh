cd /your/path/to/this/project

models=("esm1_t34_670M_UR50S" "esm1_t34_670M_UR50D" "esm1_t34_670M_UR100" "esm1_t12_85M_UR50S" "esm1_t6_43M_UR50S" "esm1b_t33_650M_UR50S" "esm1v_t33_650M_UR90S_1" "esm1v_t33_650M_UR90S_2" "esm1v_t33_650M_UR90S_3" "esm1v_t33_650M_UR90S_4" "esm1v_t33_650M_UR90S_5" "esm2_t6_8M_UR50D" "esm2_t12_35M_UR50D" "esm2_t30_150M_UR50D" "esm2_t33_650M_UR50D" "esm2_t36_3B_UR50D" "esm2_t48_15B_UR50D")
strategies=("esm1v_1" "esm1v_2" "esm1v_3" "esm1v_4" "esm1v_5" "AR_1" "AR_2" "AR_3" "AR_4")
for model in "${models[@]}"
do
    for strategy in "${strategies[@]}"
    do
        python rank_all_dms.py --dms-dir data/dms --model-name $model --rank-strategy $strategy
    done
done