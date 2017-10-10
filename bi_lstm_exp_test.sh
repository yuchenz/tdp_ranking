data_dir=.
train_file=tmp.txt
test_file=tmp.txt

python -u train.py --train_file $data_dir/${train_file} --model_file models/${train_file}.bilstm-model --iter 10 --classifier bi_lstm --labeled 

if [ -f $data_dir/${train_file}.bilstm-parsed-l ]; 
then
    mv $data_dir/${train_file}.bilstm-parsed-l ~/.recycle
fi

python parse.py --test_file $data_dir/${test_file} --model_file models/${train_file}.bilstm-model --vocab_file models/${train_file}.vocab --parsed_file $data_dir/${test_file}.bilstm-parsed --classifier bi_lstm --labeled

python eval.py --gold_file $data_dir/${test_file} --parsed_file $data_dir/${test_file}.bilstm-parsed --labeled 
