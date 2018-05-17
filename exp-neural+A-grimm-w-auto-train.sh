##########################
##  parameter settings  ## 
##########################

exp_id=$1
iter=$2
labeled=$3

data_dir=../all_annotated_data

train_train_file=$data_dir/grimm.cross-valid-train.stage1_output.20180516-0.gold_edge_added
train_dev_file=$data_dir/grimm.dev.stage1_output.20180516-0.gold_edge_added

test_dev_file=$data_dir/grimm.dev.stage1_output.20180516-0
test_test_file=$data_dir/grimm.test.stage1_output.20180516-0

eval_dev_file=$data_dir/grimm.dev
eval_test_file=$data_dir/grimm.test

model_file=models/grimm.cross-valid-train.neural-model.$exp_id 
vocab_file=models/${train_file_stem}.neural-model.${exp_id}.vocab 


################
##  training  ##  
################

echo training ...
python -u train.py --train_file $train_file --dev_file $dev_file --model_file $model_file --iter $iter --classifier bi_lstm $labeled 


###########################################
##  parse and evaluate on test_dev data  ## 
###########################################

echo parsing dev data ...
if [ -f ${test_dev_file}.stage2-neural-parsed-$l.$exp_id ]; 
then
    mv ${test_dev_file}.stage2-neural-parsed-$l.$exp_id ~/.recycle
fi

python parse.py --test_file $test_dev_file --model_file $model_file --vocab_file $vocab_file --parsed_file ${test_dev_file}.stage2-neural-parsed-$l.$exp_id --classifier bi_lstm $labeled

echo eval dev data ...
python eval.py --gold_file $eval_dev_file --parsed_file ${test_dev_file}.stage2-neural-parsed-$l.$exp_id $labeled 


############################################
##  parse and evaluate on test_test data  ## 
############################################

echo parsing test data ...
if [ -f ${test_test_file}.stage2-neural-parsed-$l.$exp_id ]; 
then
    mv ${test_test_file}.stage2-neural-parsed-$l.$exp_id ~/.recycle
fi

python parse.py --test_file $test_test_file --model_file $model_file --vocab_file $vocab_file --parsed_file ${test_test_file}.stage2-neural-parsed-$l.$exp_id --classifier bi_lstm $labeled

echo eval test data ...
python eval.py --gold_file $eval_test_file --parsed_file ${test_test_file}.stage2-neural-parsed-$l.$exp_id $labeled 


#############################################
##  for models trained with labeled data,  ##
##    do unlabeled evaluations too         ## 
#############################################

if [ $labeled == "--labeled" ];
then
    echo unlabeled evaluations on dev data ...
    python eval.py --gold_file $eval_dev_file --parsed_file ${test_dev_file}.stage2-neural-parsed-$l.$exp_id
    echo unlabeled evaluations on test data ...
    python eval.py --gold_file $eval_test_file --parsed_file ${test_test_file}.stage2-neural-parsed-$l.$exp_id
fi
