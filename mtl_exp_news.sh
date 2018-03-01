data_dir=../all_annotated_data
train_file=all_news.train
dev_file=all_news.dev
test_file=all_news.test
exp_id=$1
iter=$2
labeled=$3

if [ $labeled == "--labeled" ];
then
    l_ul=-l
else
    l_ul=-ul
fi

echo $exp_id $iter $labeled

echo ========================================================
echo training ...
python -u train.py --train_file $data_dir/${train_file} --dev_file $data_dir/$dev_file --model_file models/${train_file}.mtl-model.$exp_id --iter $iter --classifier mtl $labeled 

echo ========================================================
echo parsing training data ...
python parse.py --test_file $data_dir/${train_file} --model_file models/${train_file}.mtl-model.$exp_id --vocab_file models/${train_file}.mtl-model.${exp_id}.vocab --parsed_file $data_dir/${train_file}.mtl-parsed$l_ul.$exp_id --classifier mtl $labeled

echo ========================================================
echo eval ...
python eval.py --gold_file $data_dir/${train_file} --parsed_file ${data_dir}/${train_file}.mtl-parsed$l_ul.$exp_id $labeled

echo ========================================================
echo parsing dev data ...
python parse.py --test_file $data_dir/${dev_file} --model_file models/${train_file}.mtl-model.$exp_id --vocab_file models/${train_file}.mtl-model.${exp_id}.vocab --parsed_file $data_dir/${dev_file}.mtl-parsed$l_ul.$exp_id --classifier mtl $labeled

echo ========================================================
echo eval ...
python eval.py --gold_file $data_dir/${dev_file} --parsed_file ${data_dir}/${dev_file}.mtl-parsed$l_ul.$exp_id $labeled

echo ========================================================
echo parsing test data ...
python parse.py --test_file $data_dir/${test_file} --model_file models/${train_file}.mtl-model.$exp_id --vocab_file models/${train_file}.mtl-model.${exp_id}.vocab --parsed_file $data_dir/${test_file}.mtl-parsed$l_ul.$exp_id --classifier mtl $labeled

echo ========================================================
echo eval ...
python eval.py --gold_file $data_dir/${test_file} --parsed_file ${data_dir}/${test_file}.mtl-parsed$l_ul.$exp_id $labeled
