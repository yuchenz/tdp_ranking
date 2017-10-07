
data_dir=../all_annotated_data

data=$1
model=$2
iter=$3
classifier=$4
labeled=$5

echo $data $model 

echo python -u train.py --train_file $data_dir/${data}.train --model_file models/${data}_train.${model}_model --iter $iter --classifier $classifier $labeled
python -u train.py --train_file $data_dir/${data}.train --model_file models/${data}_train.${model}_model --iter $iter --classifier $classifier $labeled

: <<'END'
if [ -f $data_dir/${data}.dev.parsed ]; 
then
    echo mv $data_dir/${data}.dev.parsed ~/.recycle
    mv $data_dir/${data}.dev.parsed ~/.recycle
fi

if [ -f $data_dir/${data}.train.parsed ]; 
then
    echo mv $data_dir/${data}.train.parsed ~/.recycle
    mv $data_dir/${data}.train.parsed ~/.recycle
fi

echo python parse.py $data_dir/${data}.dev models/${data}_train.${model}_model models/${data}_train.vocab $data_dir/${data}.dev.${model}.parsed lr $labeled
python parse.py $data_dir/${data}.dev models/${data}_train.${model}_model models/${data}_train.vocab $data_dir/${data}.dev.${model}.parsed lr $labeled
echo python parse.py $data_dir/${data}.train models/${data}_train.${model}_model models/${data}_train.vocab $data_dir/${data}.train.${model}.parsed lr $labeled
python parse.py $data_dir/${data}.train models/${data}_train.${model}_model models/${data}_train.vocab $data_dir/${data}.train.${model}.parsed lr $labeled

echo python eval.py $data_dir/${data}.dev $data_dir/${data}.dev.${model}.parsed $labeled
python eval.py $data_dir/${data}.dev $data_dir/${data}.dev.${model}.parsed $labeled

echo python eval.py $data_dir/${data}.train $data_dir/${data}.train.${model}.parsed $labeled
python eval.py $data_dir/${data}.train $data_dir/${data}.train.${model}.parsed $labeled
END
