##########################
##  parameter settings  ## 
##########################

exp_id=$1
iter=$2

data_dir=../crowd_sourcing/crowd_sourced_data/

train_file_stem=timebank-dense.yuchen-tdt.train
dev_file_stem=timebank-dense.yuchen-tdt.dev
train_file=$data_dir/$train_file_stem
dev_file=$data_dir/$dev_file_stem

test_dev_file=$data_dir/timebank-dense.yuchen-tdt.dev
test_test_file=$data_dir/timebank-dense.yuchen-tdt.test

eval_dev_file=$data_dir/timebank-dense.yuchen-tdt.dev
eval_test_file=$data_dir/timebank-dense.yuchen-tdt.test

labeled=$3

if [ $labeled == "--labeled" ];
then
    l=labeled
else
    l=unlabeled
fi

echo exp_id $exp_id
echo iter $iter
echo data_dir $data_dir
echo train_file $train_file
echo dev_file $dev_file
echo test_train_file $train_file
echo test_dev_file $test_dev_file
echo test_test_file $test_test_file
echo labeled $labeled

model_file=models/${train_file_stem}.neural-model.$exp_id 
vocab_file=models/${train_file_stem}.neural-model.${exp_id}.vocab 

################
##  training  ##  
################

echo training ...
python -u train.py --train_file $train_file --dev_file $dev_file --model_file $model_file --iter $iter --classifier bi_lstm $labeled 

#: <<'END'
###########################################
##  parse and evaluate on training data  ##  
###########################################

echo parsing training data ...
if [ -f ${train_file}.parsed-tdt-$l.$exp_id ]; 
then
    mv ${train_file}.parsed-tdt-$l.$exp_id ~/.recycle
fi

python parse.py --test_file $train_file --model_file $model_file --vocab_file $vocab_file --parsed_file ${train_file}.parsed-tdt-$l.$exp_id --classifier bi_lstm $labeled

echo eval training data ...
python eval.py --gold_file $train_file --parsed_file ${train_file}.parsed-tdt-$l.$exp_id $labeled 


###########################################
##  parse and evaluate on test_dev data  ## 
###########################################

echo parsing dev data ...
if [ -f ${test_dev_file}.parsed-tdt-$l.$exp_id ]; 
then
    mv ${test_dev_file}.parsed-tdt-$l.$exp_id ~/.recycle
fi

python parse.py --test_file $test_dev_file --model_file $model_file --vocab_file $vocab_file --parsed_file ${test_dev_file}.parsed-tdt-$l.$exp_id --classifier bi_lstm $labeled

echo eval dev data ...
python eval.py --gold_file $eval_dev_file --parsed_file ${test_dev_file}.parsed-tdt-$l.$exp_id $labeled 


############################################
##  parse and evaluate on test_test data  ## 
############################################

echo parsing test data ...
if [ -f ${test_test_file}.parsed-tdt-$l.$exp_id ]; 
then
    mv ${test_test_file}.parsed-tdt-$l.$exp_id ~/.recycle
fi

python parse.py --test_file $test_test_file --model_file $model_file --vocab_file $vocab_file --parsed_file ${test_test_file}.parsed-tdt-$l.$exp_id --classifier bi_lstm $labeled

echo eval test data ...
python eval.py --gold_file $eval_test_file --parsed_file ${test_test_file}.parsed-tdt-$l.$exp_id $labeled 


#############################################
##  for models trained with labeled data,  ##
##    do unlabeled evaluations too         ## 
#############################################

if [ $labeled == "--labeled" ];
then
    echo unlabeled evaluations on training data ...
    python eval.py --gold_file $train_file --parsed_file ${train_file}.parsed-tdt-$l.$exp_id
    echo unlabeled evaluations on dev data ...
    python eval.py --gold_file $eval_dev_file --parsed_file ${test_dev_file}.parsed-tdt-$l.$exp_id
    echo unlabeled evaluations on test data ...
    python eval.py --gold_file $eval_test_file --parsed_file ${test_test_file}.parsed-tdt-$l.$exp_id
fi
#END
