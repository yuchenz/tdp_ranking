data_dir=.
train_file=tmp.txt
test_file=tmp.txt
exp_id=$1
iter=$2

echo training ...
python -u train.py --train_file $data_dir/${train_file} --model_file models/${train_file}.bilstm-model.$exp_id --iter $iter --classifier bi_lstm --labeled 

echo parsing training data ...
if [ -f $data_dir/${train_file}.bilstm-parsed-l.$exp_id ]; 
then
    mv $data_dir/${train_file}.bilstm-parsed-l.$exp_id ~/.recycle
fi

python parse.py --test_file $data_dir/${train_file} --model_file models/${train_file}.bilstm-model.$exp_id --vocab_file models/${train_file}.bilstm-model.${exp_id}.vocab --parsed_file $data_dir/${train_file}.bilstm-parsed-l.$exp_id --classifier bi_lstm --labeled

echo eval ...
python eval.py --gold_file $data_dir/${train_file} --parsed_file $data_dir/${train_file}.bilstm-parsed-l.$exp_id --labeled 
