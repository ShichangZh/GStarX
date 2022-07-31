# +
# python run_gstarx.py models='gcn' datasets='bace' explainers.sparsity=0.8 device_id=0

# +
# python run_gstarx.py models=$1 datasets=$2 explainers.sparsity=$3 device_id=$4

# +
## Uncomment the code below for multi-sparsity experiment
# $1 model
# $2 dataset
# $3 device

# method="gstarx-$1"
# output_file="output/$method.txt" 
# for sp in 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 
# do
#     printf "%s," $method >> $output_file
#     printf "%s," $2 >> $output_file
#     python run_gstarx.py models=$1 datasets=$2 device_id=$3 \
#                                explainers.sparsity=$th >> $output_file        
# done



# +
# Uncomment the code below for multi-dataset multi-sparsity experiment
# $1 model
# $2 device

method="gstarx-$1"
output_file="output/$method.txt" 
for ds in "ba_2motifs" "bace" "bbbp" "graph_sst2" "mutag" "twitter"
do
    for sp in 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75
    do
        printf "%s," $method >> $output_file
        printf "%s," $ds >> $output_file
        python run_gstarx.py models=$1 datasets=$ds device_id=$2 \
                                explainers.sparsity=$sp >> $output_file        
    done
done
