data_dir=../all_annotated_data
train_file=all_news.train
test_file=all_news.dev

echo training ...
python -u train.py --train_file $data_dir/${train_file} --model_file models/${train_file}.bilstm-model --iter 100 --classifier bi_lstm --labeled 

echo parsing ...
if [ -f $data_dir/${train_file}.bilstm-parsed-l ]; 
then
    mv $data_dir/${train_file}.bilstm-parsed-l ~/.recycle
fi

python parse.py --test_file $data_dir/${test_file} --model_file models/${train_file}.bilstm-model --vocab_file models/${train_file}.vocab --parsed_file $data_dir/${test_file}.bilstm-parsed --classifier bi_lstm --labeled

echo eval ...
python eval.py --gold_file $data_dir/${test_file} --parsed_file $data_dir/${test_file}.bilstm-parsed --labeled 
