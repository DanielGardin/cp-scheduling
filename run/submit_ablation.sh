for frac in 0.1 0.25 0.33 0.5 0.8 ; do
    for i in $(seq 1 10);
    do
        python run/pl_behavior_cloning.py \
            --seed $i \
            --frac_train $frac \
            --lr 1e-5 \
            -q \
            --log_dir "logs/ablations/frac_train"
        echo "Finished iteration $i with frac_train $frac"
    done
done


for frac in 0.1 0.25 0.33 0.5 0.8 ; do
    for i in $(seq 1 10);
    do
        python run/pl_reinforce.py \
            --seed $i \
            --frac_train $frac \
            --lr 1e-5 \
            -q \
            --log_dir "logs/ablations/reinforce_frac"
        echo "Finished iteration $i with frac_train $frac"
    done
done
