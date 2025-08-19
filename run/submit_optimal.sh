for i in $(seq 1 10);
do
    python run/pl_reinforce.py \
        --seed $i \
        --lr 1e-5 \
        --log_dir "logs/optimal/reinforce" \
        --optimal \
        --steps 1000000
    echo "Finished iteration $i"
done