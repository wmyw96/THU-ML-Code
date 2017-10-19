#for learnRate in 0.005 0.01 0.02 0.05; do
for learnRate in 0.01; do
    #for reLambda in 0.01 0.05 0.1 0.2 0.5; do
    for reLambda in 0.1; do
        for droupout in 0.3 0.4 0.5 0.6 0.7; do
            echo ${learnRate} ${reLambda} ${dropout}
            python pmf_vi.py ${learnRate} ${reLambda} ${dropout} > log/pmf_vi_${learnRate}_${reLambda}_${dropout}.log &
        done
    done
done
