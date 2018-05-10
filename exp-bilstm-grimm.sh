data_dir=../all_annotated_data
train_train_file=grimm.train
train_dev_file=grimm.dev

test_dev_file=grimm.dev.stage1-out-20180509-3
test_test_file=grimm.test.stage1-out-20180509-3

eval_dev_file=$train_dev_file
eval_test_file=grimm.test

exp_id=$1
iter=$2
labeled=$3

echo training ...
python -u train.py --train_file $data_dir/${train_train_file} --dev_file $data_dir/$train_dev_file --model_file models/${train_train_file}.bilstm-model.$exp_id --iter $iter --classifier bi_lstm --timex_event_label_input timex_event $labeled 

: <<'END'
echo parsing training data ...
if [ -f $data_dir/${train_train_file}.stage2-out-$exp_id ]; 
then
    mv $data_dir/${train_train_file}.stage2-out-$exp_id ~/.recycle
fi

python parse.py --test_file $data_dir/${train_train_file} --model_file models/${train_train_file}.bilstm-model.$exp_id --vocab_file models/${train_train_file}.bilstm-model.${exp_id}.vocab --parsed_file $data_dir/${train_train_file}.stage2-out-$exp_id --classifier bi_lstm --timex_event_label_input timex_event $labeled

echo eval ...
python eval.py --gold_file $data_dir/${train_train_file} --parsed_file $data_dir/${train_train_file}.stage2-out-$exp_id $labeled 
END

echo parsing dev data ...
if [ -f $data_dir/${test_dev_file}.stage2-out-$exp_id ]; 
then
    mv $data_dir/${test_dev_file}.stage2-out-$exp_id ~/.recycle
fi

python parse.py --test_file $data_dir/${test_dev_file} --model_file models/${train_train_file}.bilstm-model.$exp_id --vocab_file models/${train_train_file}.bilstm-model.${exp_id}.vocab --parsed_file $data_dir/${test_dev_file}.stage2-out-$exp_id --classifier bi_lstm --timex_event_label_input timex_event $labeled

echo eval ...
python eval.py --gold_file $data_dir/${eval_dev_file} --parsed_file $data_dir/${test_dev_file}.stage2-out-$exp_id $labeled 


echo parsing test data ...
if [ -f $data_dir/${test_test_file}.stage2-out-$exp_id ]; 
then
    mv $data_dir/${test_test_file}.stage2-out-$exp_id ~/.recycle
fi

python parse.py --test_file $data_dir/${test_test_file} --model_file models/${train_train_file}.bilstm-model.$exp_id --vocab_file models/${train_train_file}.bilstm-model.${exp_id}.vocab --parsed_file $data_dir/${test_test_file}.stage2-out-$exp_id --classifier bi_lstm --timex_event_label_input timex_event $labeled

echo eval ...
python eval.py --gold_file $data_dir/${eval_test_file} --parsed_file $data_dir/${test_test_file}.stage2-out-$exp_id $labeled 

if [ $labeled == "--labeled" ];
then
    echo unlabeled eval ...
    #python eval.py --gold_file $data_dir/${train_train_file} --parsed_file $data_dir/${train_train_file}.stage2-out-$exp_id
    python eval.py --gold_file $data_dir/${eval_dev_file} --parsed_file $data_dir/${test_dev_file}.stage2-out-$exp_id
    python eval.py --gold_file $data_dir/${eval_test_file} --parsed_file $data_dir/${test_test_file}.stage2-out-$exp_id
fi
