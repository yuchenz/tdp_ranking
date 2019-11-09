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

train_file_stem=timebank-dense.yuchen-tdt.train
dev_file_stem=timebank-dense.yuchen-tdt.dev
train_file=$data_dir/$train_file_stem
dev_file=$data_dir/$dev_file_stem

train_file1_stem=timebank-dense.yuchen-tdt.train
dev_file1_stem=timebank-dense.yuchen-tdt.dev
test_file1_stem=timebank-dense.yuchen-tdt.test

train_file1=$data_dir/$train_file1_stem
dev_file1=$data_dir/$dev_file1_stem
test_file1=$data_dir/$test_file1_stem

train_file2_stem=timebank-dense.turker-tdt.train
dev_file2_stem=timebank-dense.turker-tdt.dev
test_file2_stem=timebank-dense.turker-tdt.test

train_file2=$data_dir/$train_file2_stem
dev_file2=$data_dir/$dev_file2_stem
test_file2=$data_dir/$test_file2_stem

echo exp_id $exp_id
echo iter $iter
echo data_dir $data_dir
echo train_file $train_file
echo dev_file $dev_file
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

echo parsing gold training data ...
if [ -f ${train_file1}.parsed-tdt.$exp_id ]; 
then
    mv ${train_file1}.parsed-tdt.$exp_id ~/.recycle
fi

python parse.py --test_file $train_file1 --model_file $model_file --vocab_file $vocab_file --parsed_file ${train_file1}.parsed-tdt.$exp_id --classifier log_reg $labeled

echo eval on gold training data ...
python eval.py --gold_file $train_file1 --parsed_file ${train_file1}.parsed-tdt.$exp_id $labeled 
echo


echo parsing crowd training data ...
if [ -f ${train_file2}.parsed-tdt.$exp_id ]; 
then
    mv ${train_file2}.parsed-tdt.$exp_id ~/.recycle
fi

python parse.py --test_file $train_file2 --model_file $model_file --vocab_file $vocab_file --parsed_file ${train_file2}.parsed-tdt.$exp_id --classifier log_reg $labeled

echo eval on crowd training data ...
python eval.py --gold_file $train_file2 --parsed_file ${train_file2}.parsed-tdt.$exp_id $labeled 
echo


###########################################
##  parse and evaluate on dev data  ## 
###########################################

echo parsing gold dev data ...
if [ -f ${dev_file1}.parsed-tdt.$exp_id ]; 
then
    mv ${dev_file1}.parsed-tdt.$exp_id ~/.recycle
fi

python parse.py --test_file $dev_file1 --model_file $model_file --vocab_file $vocab_file --parsed_file ${dev_file1}.parsed-tdt.$exp_id --classifier log_reg $labeled

echo eval gold dev data ...
python eval.py --gold_file $dev_file1 --parsed_file ${dev_file1}.parsed-tdt.$exp_id $labeled 
echo


echo parsing crowd dev data ...
if [ -f ${dev_file2}.parsed-tdt.$exp_id ]; 
then
    mv ${dev_file2}.parsed-tdt.$exp_id ~/.recycle
fi

python parse.py --test_file $dev_file2 --model_file $model_file --vocab_file $vocab_file --parsed_file ${dev_file2}.parsed-tdt.$exp_id --classifier log_reg $labeled

echo eval crowd dev data ...
python eval.py --gold_file $dev_file2 --parsed_file ${dev_file2}.parsed-tdt.$exp_id $labeled 
echo


############################################
##  parse and evaluate on test data  ## 
############################################

echo parsing gold test data ...
if [ -f ${test_file1}.parsed-tdt.$exp_id ]; 
then
    mv ${test_file1}.parsed-tdt.$exp_id ~/.recycle
fi

python parse.py --test_file $test_file1 --model_file $model_file --vocab_file $vocab_file --parsed_file ${test_file1}.parsed-tdt.$exp_id --classifier log_reg $labeled

echo eval gold test data ...
python eval.py --gold_file $test_file1 --parsed_file ${test_file1}.parsed-tdt.$exp_id $labeled 
echo


echo parsing crowd test data ...
if [ -f ${test_file2}.parsed-tdt.$exp_id ]; 
then
    mv ${test_file2}.parsed-tdt.$exp_id ~/.recycle
fi

python parse.py --test_file $test_file2 --model_file $model_file --vocab_file $vocab_file --parsed_file ${test_file2}.parsed-tdt.$exp_id --classifier log_reg $labeled

echo eval crowd test data ...
python eval.py --gold_file $test_file2 --parsed_file ${test_file2}.parsed-tdt.$exp_id $labeled 
echo


#############################################
##  for models trained with labeled data,  ##
##    do unlabeled evaluations too         ## 
#############################################

if [ $labeled == "--labeled" ];
then
    echo unlabeled evaluations on gold training data ...
    python eval.py --gold_file $train_file1 --parsed_file ${train_file1}.parsed-tdt.$exp_id
    echo
    echo unlabeled evaluations on crowd training data ...
    python eval.py --gold_file $train_file2 --parsed_file ${train_file2}.parsed-tdt.$exp_id
    echo

    echo unlabeled evaluations on gold dev data ...
    python eval.py --gold_file $dev_file1 --parsed_file ${dev_file1}.parsed-tdt.$exp_id
    echo
    echo unlabeled evaluations on crowd dev data ...
    python eval.py --gold_file $dev_file2 --parsed_file ${dev_file2}.parsed-tdt.$exp_id
    echo

    echo unlabeled evaluations on gold test data ...
    python eval.py --gold_file $test_file1 --parsed_file ${test_file1}.parsed-tdt.$exp_id
    echo
    echo unlabeled evaluations on crowd test data ...
    python eval.py --gold_file $test_file2 --parsed_file ${test_file2}.parsed-tdt.$exp_id
    echo
fi
#END
