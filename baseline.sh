test_test_file=$1
eval_test_file=$2
default_label=$3

if [ -f ${test_test_file}.baseline-parsed-ul ]; 
then
    mv ${test_test_file}.baseline-parsed-ul ~/.recycle
fi

echo unlabeled ...
python parse.py --test_file $test_test_file --classifier baseline --parsed_file ${test_test_file}.baseline-parsed-ul --default_label $default_label
python eval.py --gold_file $eval_test_file --parsed_file ${test_test_file}.baseline-parsed-ul

if [ -f ${test_file}.baseline-parsed-l ]; 
then
    mv ${test_file}.baseline-parsed-l ~/.recycle
fi

echo labeled ...
python parse.py --test_file $test_test_file --classifier baseline --parsed_file ${test_test_file}.baseline-parsed-l --default_label $default_label --labeled
python eval.py --gold_file $eval_test_file --parsed_file ${test_test_file}.baseline-parsed-l --labeled 
