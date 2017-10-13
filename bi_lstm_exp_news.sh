data_dir=../all_annotated_data
train_file=all_news.train
dev_file=all_news.dev
test_file=all_news.test
exp_id=$1
iter=$2

echo training ...
python -u train.py --train_file $data_dir/${train_file} --dev_file $data_dir/$dev_file --model_file models/${train_file}.bilstm-model.$exp_id --iter $iter --classifier bi_lstm --labeled 

echo parsing training data ...
if [ -f $data_dir/${train_file}.bilstm-parsed-l.$exp_id ]; 
then
    mv $data_dir/${train_file}.bilstm-parsed-l.$exp_id ~/.recycle
fi

python parse.py --test_file $data_dir/${train_file} --model_file models/${train_file}.bilstm-model.$exp_id --vocab_file models/${train_file}.bilstm-model.${exp_id}.vocab --parsed_file $data_dir/${train_file}.bilstm-parsed-l.$exp_id --classifier bi_lstm --labeled

echo eval ...
python eval.py --gold_file $data_dir/${train_file} --parsed_file $data_dir/${train_file}.bilstm-parsed-l.$exp_id --labeled 

echo parsing dev data ...
if [ -f $data_dir/${dev_file}.bilstm-parsed-l.$exp_id ]; 
then
    mv $data_dir/${dev_file}.bilstm-parsed-l.$exp_id ~/.recycle
fi

python parse.py --test_file $data_dir/${dev_file} --model_file models/${train_file}.bilstm-model.$exp_id --vocab_file models/${train_file}.bilstm-model.${exp_id}.vocab --parsed_file $data_dir/${dev_file}.bilstm-parsed-l.$exp_id --classifier bi_lstm --labeled

echo eval ...
python eval.py --gold_file $data_dir/${dev_file} --parsed_file $data_dir/${dev_file}.bilstm-parsed-l.$exp_id --labeled 


echo parsing test data ...
if [ -f $data_dir/${test_file}.bilstm-parsed-l.$exp_id ]; 
then
    mv $data_dir/${test_file}.bilstm-parsed-l.$exp_id ~/.recycle
fi

python parse.py --test_file $data_dir/${test_file} --model_file models/${train_file}.bilstm-model.$exp_id --vocab_file models/${train_file}.bilstm-model.${exp_id}.vocab --parsed_file $data_dir/${test_file}.bilstm-parsed-l.$exp_id --classifier bi_lstm --labeled

echo eval ...
python eval.py --gold_file $data_dir/${test_file} --parsed_file $data_dir/${test_file}.bilstm-parsed-l.$exp_id --labeled 
