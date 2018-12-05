data_dir=../crowd_sourcing/crowd_sourced_data/
train_train_file=timebank-all_tb-dense-train.tdt
train_dev_file=timebank-dense.yuchen-tdt.dev

test_dev_file=timebank-dense.yuchen-tdt.dev
test_test_file=timebank-dense.yuchen-tdt.test

eval_dev_file=$train_dev_file
eval_test_file=timebank-dense.yuchen-tdt.test

exp_id=$1
iter=$2
labeled=$3

echo training ...
python -u train.py --train_file $data_dir/${train_train_file} --dev_file $data_dir/$train_dev_file --model_file models/${train_train_file}.logreg-model-$exp_id --iter $iter --classifier log_reg $labeled 

echo parsing dev data ...
if [ -f $data_dir/${test_dev_file}.parsed-tdt-$exp_id ]; 
then
    mv $data_dir/${test_dev_file}.parsed-tdt-$exp_id ~/.recycle
fi

python parse.py --test_file $data_dir/${test_dev_file} --model_file models/${train_train_file}.logreg-model-$exp_id --vocab_file models/${train_train_file}.logreg-model-${exp_id}.vocab --parsed_file $data_dir/${test_dev_file}.parsed-tdt-$exp_id --classifier log_reg $labeled

echo eval ...
python eval.py --gold_file $data_dir/${eval_dev_file} --parsed_file $data_dir/${test_dev_file}.parsed-tdt-$exp_id $labeled 

echo parsing test data ...
if [ -f $data_dir/${test_test_file}.parsed-tdt-$exp_id ]; 
then
    mv $data_dir/${test_test_file}.parsed-tdt-$exp_id ~/.recycle
fi

python parse.py --test_file $data_dir/${test_test_file} --model_file models/${train_train_file}.logreg-model-$exp_id --vocab_file models/${train_train_file}.logreg-model-${exp_id}.vocab --parsed_file $data_dir/${test_test_file}.parsed-tdt-$exp_id --classifier log_reg $labeled

echo eval ...
python eval.py --gold_file $data_dir/${eval_test_file} --parsed_file $data_dir/${test_test_file}.parsed-tdt-$exp_id $labeled 

if [ $labeled == '--labeled' ];
then
    echo unlabeled eval ...
    python eval.py --gold_file $data_dir/${eval_dev_file} --parsed_file $data_dir/${test_dev_file}.parsed-tdt-$exp_id
    python eval.py --gold_file $data_dir/${eval_test_file} --parsed_file $data_dir/${test_test_file}.parsed-tdt-$exp_id 
fi
