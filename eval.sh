#!/bin/bash

#e.g. bash ./src/eval.sh yago_ko_0.4 0 1 -1 yago 7

time_start=$(date +"%Y%m%d_%H%M%S")

green=`tput setaf 2`
red=`tput setaf 1`
yellow=`tput setaf 3`
reset=`tput sgr0`

# input variables
model=${1:-HEER} # which model to run: HEER, DistMult, ComplEx, ProjE, ConvE, node2vec
entity_pred=${2:-true}
relation_pred=${3:-true}
gpu=${2:-0} # working gpu for prediction

# find relative root directory
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
script_dir="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
root_dir="$( dirname $script_dir )"/network-embedding-evaluation


eval_file="$root_dir"/datasets/FB15K237/fb15k237_ko_0.07_eval.txt

per_type_cur_model_dir="$root_dir"/datasets/FB15K237/per_type_temp/ #TODO: take a model name
mkdir -p "$per_type_cur_model_dir"

echo ${yellow}===$model Testing===${reset}
#echo "Splitting test cases by edge type..."
#python2 "$root_dir"/util/separate_edges_by_types.py --input-file=$eval_file --output-dir="$per_type_cur_model_dir"
#echo "Done."

output_file="$root_dir"/results/"$model"_metrics.txt

tf_models=("ProjE")
pt_models=("HEER DistMult ComplEx")
echo "Computing entity predictions for all test cases..."
if [[ "${pt_models[@]}" =~ "${model}" ]]; then
  if [[ "${entity_pred}" =~ "true" ]]; then
	  python2 "$root_dir"/pred_head_tail.py --model=$model --batch-size=128 --gpu=$gpu --test-dir="$per_type_cur_model_dir"
  fi
  if [[ "${relation_pred}" =~ "true" ]]; then
    python2 "$root_dir"/pred_relation.py --model=$model --batch-size=128 --gpu=$gpu --test-dir="$per_type_cur_model_dir"
  fi
else
  if [[ "${entity_pred}" =~ "true" ]]; then
	  python3 "$root_dir"/pred_head_tail_tf.py --model=$model --batch-size=128 --gpu=$gpu --test-dir="$per_type_cur_model_dir"
  fi
  if [[ "${relation_pred}" =~ "true" ]]; then
    python3 "$root_dir"/pred_relation_tf.py --model=$model --batch-size=128 --gpu=$gpu --test-dir="$per_type_cur_model_dir"
  fi
fi
echo "Done."

#score_file="$root_dir"/results/"$model"_scores.txt

#echo "Merging edge type prediction scores..."
#python2 "$root_dir"/util/merge_edges_with_all_types.py --input-ref-file $eval_file --input-score-dir "$per_type_cur_model_dir" --input-score-keywords _pred --output-file "$score_file"
#echo "Done."

#echo "Computing MRR from scores..."
#python3 "$root_dir"/util/mrr_from_score.py --input-score-file $score_file --input-eval-file $eval_file > "$output_file"
#echo "Done."

echo "Cleaning up..."
rm -r "$per_type_cur_model_dir"
