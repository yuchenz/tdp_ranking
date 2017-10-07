python -u train.py --train_file tmp.txt --model_file tmp.txt.bilstm_model --iter 10 --classifier bi_lstm --labeled 

if [ -f tmp.txt.bilstm_parsed ]; 
then
    mv tmp.txt.bilstm_parsed ~/.recycle
fi

python parse.py --test_file tmp.txt --model_file tmp.txt.bilstm_model --vocab_file models/tmp_txt.vocab --parsed_file tmp.txt.bilstm-parsed --classifier bi_lstm --labeled

python eval.py --gold_file tmp.txt --parsed_file tmp.txt.bilstm-parsed --labeled 
