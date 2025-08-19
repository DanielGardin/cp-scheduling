for pdr in "WSPT" "WMDD" "COverT" "ATC" ; do
    for frac in 0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0 ; do
        python run/pl_parameter_estimation.py \
            --seed 0 \
            --pdr $pdr \
            --lr 3e-5 \
            --frac_train $frac \
            -q \
            --device cpu \
            --log_dir "logs/parameter_ablation/$pdr" &
    done
done
