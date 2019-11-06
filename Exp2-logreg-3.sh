##########################
##  parameter settings  ## 
##########################

exp_id=$1
iter=$2

labeled=$3

if [ $labeled == "--labeled" ];
then
    l=labeled
else
    l=unlabeled
fi

data_dir=../crowd_sourcing/crowd_sourced_data/

train_file_stem=timebank-dense.turker-tdt.train
dev_file_stem=timebank-dense.turker-tdt.dev
test_file_stem=timebank-dense.turker-tdt.test

train_file=$data_dir/$train_file_stem
dev_file=$data_dir/$dev_file_stem
test_file=$data_dir/$test_file_stem

echo exp_id $exp_id
echo iter $iter
echo data_dir $data_dir
echo train_file $train_file
echo dev_file $dev_file
echo test_file $test_file
echo labeled $labeled

model_file=models/${train_file_stem}.logreg-model.$exp_id 
vocab_file=models/${train_file_stem}.logreg-model.${exp_id}.vocab 

################
##  training  ##  
################

echo training ...
python -u train.py --train_file $train_file --dev_file $dev_file --model_file $model_file --iter $iter --classifier log_reg $labeled 

#: <<'END'
###########################################
##  parse and evaluate on training data  ##  
###########################################

echo parsing training data ...
if [ -f ${train_file}.parsed-tdt.$exp_id ]; 
then
    mv ${train_file}.parsed-tdt.$exp_id ~/.recycle
fi

python parse.py --test_file $train_file --model_file $model_file --vocab_file $vocab_file --parsed_file ${train_file}.parsed-tdt.$exp_id --classifier log_reg $labeled

echo eval training data ...
python eval.py --gold_file $train_file --parsed_file ${train_file}.parsed-tdt.$exp_id $labeled 


###########################################
##  parse and evaluate on test_dev data  ## 
###########################################

echo parsing dev data ...
if [ -f ${dev_file}.parsed-tdt.$exp_id ]; 
then
    mv ${dev_file}.parsed-tdt.$exp_id ~/.recycle
fi

python parse.py --test_file $dev_file --model_file $model_file --vocab_file $vocab_file --parsed_file ${dev_file}.parsed-tdt.$exp_id --classifier log_reg $labeled

echo eval dev data ...
python eval.py --gold_file $dev_file --parsed_file ${dev_file}.parsed-tdt.$exp_id $labeled 


############################################
##  parse and evaluate on test_test data  ## 
############################################

echo parsing test data ...
if [ -f ${test_file}.parsed-tdt.$exp_id ]; 
then
    mv ${test_file}.parsed-tdt.$exp_id ~/.recycle
fi

python parse.py --test_file $test_file --model_file $model_file --vocab_file $vocab_file --parsed_file ${test_file}.parsed-tdt.$exp_id --classifier log_reg $labeled

echo eval test data ...
python eval.py --gold_file $test_file --parsed_file ${test_file}.parsed-tdt.$exp_id $labeled 


#############################################
##  for models trained with labeled data,  ##
##    do unlabeled evaluations too         ## 
#############################################

if [ $labeled == "--labeled" ];
then
    echo unlabeled evaluations on training data ...
    python eval.py --gold_file $train_file --parsed_file ${train_file}.parsed-tdt.$exp_id
    echo unlabeled evaluations on dev data ...
    python eval.py --gold_file $dev_file --parsed_file ${dev_file}.parsed-tdt.$exp_id
    echo unlabeled evaluations on test data ...
    python eval.py --gold_file $test_file --parsed_file ${test_file}.parsed-tdt.$exp_id
fi
#END
