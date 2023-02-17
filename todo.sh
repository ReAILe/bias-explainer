echo "Rank is: ${OMPI_COMM_WORLD_RANK}"

ulimit -t unlimited
ulimit -c 0
shopt -s nullglob
numthreads=$((OMPI_COMM_WORLD_SIZE))
mythread=$((OMPI_COMM_WORLD_RANK))

# tlimit="2000"
memlimit="4000000"
ulimit -v $memlimit

# ========================================================================================
# Thesis experiments

# Scalability
# python3 -W ignore -u run.py --model lr svm-linear dt --encoding Learn-efficient --feature_threshold 0.25 0.5 0.75 1 --thread=$mythread --max_thread=$numthreads> data/output/$mythread:$(date +"%d-%m-%Y-%T".txt)  2>&1
# python3 -W ignore -u run.py --model dt --encoding Learn-efficient-dependency --dt_depth 5 10 20 40 80 160 200 -1 --feature_threshold 0.25 0.5 0.75 1 --thread=$mythread --max_thread=$numthreads> data/output/$mythread:$(date +"%d-%m-%Y-%T".txt)  2>&1


# Train classifier on subset of features
# python3 -W ignore -u run.py --train --model lr svm-linear dt --dt_depth 5 10 20 40 80 160 200 -1 --thread=$mythread --max_thread=$numthreads> data/output/$mythread:$(date +"%d-%m-%Y-%T".txt)  2>&1


# Compare different encodings
python3 -W ignore -u run.py  --compare_fairsquare --model dt --encoding Learn-efficient-dependency Learn-efficient verifair fairsquare--fraction_features 0.25 --thread=$mythread > data/output/$mythread:$(date +"%d-%m-%Y-%T".txt)  2>&1


# ========================================================================================

# ** Influence function
# python3 -W ignore -u run.py --single_fold --dependency_exp --model dt lr --encoding Learn Learn-dependency --thread=$mythread > data/output/$mythread:$(date +"%d-%m-%Y-%T".txt)  2>&1

# ** comparing subset-sum vs ssat on lr and svm
# python3 -W ignore -u run.py --model svm-linear lr --encoding Learn-efficient Enum Learn-efficient-dependency Enum-dependency --thread=$mythread > data/output/$mythread:$(date +"%d-%m-%Y-%T".txt)  2>&1


# ** Comparing all verifiers
# python3 -W ignore -u run.py  --compare_fairsquare --model svm-linear lr dt --encoding Learn-efficient-dependency Learn-efficient verifair fairsquare Enum-correlation --fraction_features 0.25 --thread=$mythread > data/output/$mythread:$(date +"%d-%m-%Y-%T".txt)  2>&1
# python3 -W ignore -u run.py  --compare_fairsquare --model svm-linear lr dt --encoding Learn-efficient-dependency Learn-efficient verifair fairsquare Enum-correlation --fraction_features 0.5 --thread=$mythread > data/output/$mythread:$(date +"%d-%m-%Y-%T".txt)  2>&1
# python3 -W ignore -u run.py  --compare_fairsquare --model svm-linear lr dt --encoding Learn-efficient-dependency Learn-efficient verifair fairsquare Enum-correlation --fraction_features 0.75 --thread=$mythread > data/output/$mythread:$(date +"%d-%m-%Y-%T".txt)  2>&1
# python3 -W ignore -u run.py  --compare_fairsquare --model svm-linear lr dt --encoding Learn-efficient-dependency Learn-efficient verifair fairsquare Enum-correlation --fraction_features 1 --thread=$mythread > data/output/$mythread:$(date +"%d-%m-%Y-%T".txt)  2>&1

# python3 -W ignore -u run.py --feature_threshold 0.5 --compare_fairsquare --model svm-linear lr dt --encoding Learn-efficient-dependency Learn-efficient verifair fairsquare Enum-correlation --fraction_features 0.25 --thread=$mythread > data/output/$mythread:$(date +"%d-%m-%Y-%T".txt)  2>&1
# python3 -W ignore -u run.py --feature_threshold 0.5 --compare_fairsquare --model svm-linear lr dt --encoding Learn-efficient-dependency Learn-efficient verifair fairsquare Enum-correlation --fraction_features 0.5 --thread=$mythread > data/output/$mythread:$(date +"%d-%m-%Y-%T".txt)  2>&1
# python3 -W ignore -u run.py --feature_threshold 0.5 --compare_fairsquare --model svm-linear lr dt --encoding Learn-efficient-dependency Learn-efficient verifair fairsquare Enum-correlation --fraction_features 0.75 --thread=$mythread > data/output/$mythread:$(date +"%d-%m-%Y-%T".txt)  2>&1
# python3 -W ignore -u run.py --feature_threshold 0.5 --compare_fairsquare --model svm-linear lr dt --encoding Learn-efficient-dependency Learn-efficient verifair fairsquare Enum-correlation --fraction_features 1 --thread=$mythread > data/output/$mythread:$(date +"%d-%m-%Y-%T".txt)  2>&1


# ** Post-processing algo
# python3 -W ignore -u run_postprocessing.py --dataset adult german compas --fair_algo rco cpp --model lr --protected 0 1 --encoding Learn Learn-dependency --thread=$mythread > data/output/$mythread:$(date +"%d-%m-%Y-%T".txt)  2>&1

#  ** Preprocessing algorithms
# python3 -W ignore -u run_preprocessing.py --dataset adult compas german --fair_algo rw op --model dt lr --sampling 0 --protected 0 1 --thread=$mythread > data/output/$mythread:$(date +"%d-%m-%Y-%T".txt)  2>&1


# ** fairness attack
# No parallel
# python3 run_fairness_attack.py

#  ** Sanity check
# python3 -W ignore -u run_sanity.py --thread=$mythread > data/output/$mythread:$(date +"%d-%m-%Y-%T".txt)  2>&1
# python3 -W ignore -u distribution_sanity.py --thread=$mythread  --max_thread=$numthreads> data/output/$mythread:$(date +"%d-%m-%Y-%T".txt)  2>&1



#  ** Scalability of decision tree classifiers
# python3 -W ignore -u run.py --model dt --dt_depth 5 10 20 40 80 160 --encoding Enum Learn-efficient Enum-dependency  Learn-efficient-dependency --thread=$mythread > data/output/$mythread:$(date +"%d-%m-%Y-%T".txt)  2>&1


# ** Complexity of DAG structure
# python3 -W ignore -u run.py --model lr svm-linear --encoding Learn-efficient-dependency --feature_threshold 0.25 0.5 0.75 1 --fraction_features 1 --thread=$mythread > data/output/$mythread:$(date +"%d-%m-%Y-%T".txt)  2>&1


# ** Fairness metrics verification on multiple sensitive attributes (Comparison between Justicia nd FVGM)
# python3 -W ignore -u run.py --single_fold --model lr --compute_independence_metric --compute_sufficiency_metric --compute_separation_metric --encoding Learn-efficient-dependency Learn --thread=$mythread > data/output/$mythread:$(date +"%d-%m-%Y-%T".txt)  2>&1



# For learning DAG, put --train flag
# python3 -W ignore -u run.py --dependency_exp --train --encoding Enum-dependency --thread=$mythread > data/output/$mythread:$(date +"%d-%m-%Y-%T".txt)  2>&1



# Train classifier
# python3 -W ignore -u run.py --train --model dt CNF lr svm-linear --dt_depth 5 10 20 40 80 160 --thread=$mythread > data/output/$mythread:$(date +"%d-%m-%Y-%T".txt)  2>&1

# For comparing different encodings inside Justicia
# python3 -W ignore -u run.py --single_fold --model svm-linear lr --encoding Learn-efficient Learn-efficient-correlation Learn-efficient-dependency --thread=$mythread > data/output/$mythread:$(date +"%d-%m-%Y-%T".txt)  2>&1

# All folds
# python3 -W ignore -u run.py --model CNF dt svm-linear lr --dt_depth 5 10 20 40 80 160 --encoding Enum Learn-efficient Enum-correlation Learn-efficient-correlation Enum-dependency  Learn-efficient-dependency --thread=$mythread > data/output/$mythread:$(date +"%d-%m-%Y-%T".txt)  2>&1

# Smaller
# python3 -W ignore -u run.py --model CNF  --encoding Enum Enum-correlation  --thread=$mythread > data/output/$mythread:$(date +"%d-%m-%Y-%T".txt)  2>&1

# Train classifier on subset of features
# python3 -W ignore -u run.py --train --model lr svm-linear --fraction_features 0.25 --thread=$mythread > data/output/$mythread:$(date +"%d-%m-%Y-%T".txt)  2>&1
# python3 -W ignore -u run.py --train --model lr svm-linear --fraction_features 0.5 --thread=$mythread > data/output/$mythread:$(date +"%d-%m-%Y-%T".txt)  2>&1
# python3 -W ignore -u run.py --train --model lr svm-linear --fraction_features 0.75 --thread=$mythread > data/output/$mythread:$(date +"%d-%m-%Y-%T".txt)  2>&1
# python3 -W ignore -u run.py --train --model lr svm-linear --fraction_features 1 --thread=$mythread > data/output/$mythread:$(date +"%d-%m-%Y-%T".txt)  2>&1


