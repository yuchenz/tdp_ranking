
data_dir=../all_annotated_data

data=$1
model=$2
iter=$3

echo $data $model 

echo python train.py $data_dir/${data}.train models/${data}_train.${model}_model $iter
python train.py $data_dir/${data}.train models/${data}_train.${model}_model $iter

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

echo python parse.py $data_dir/${data}.dev models/${data}_train.${model}_model models/${data}_train.vocab $data_dir/${data}.dev.parsed
python parse.py $data_dir/${data}.dev models/${data}_train.${model}_model models/${data}_train.vocab $data_dir/${data}.dev.parsed
echo python parse.py $data_dir/${data}.train models/${data}_train.${model}_model models/${data}_train.vocab $data_dir/${data}.train.parsed
python parse.py $data_dir/${data}.train models/${data}_train.${model}_model models/${data}_train.vocab $data_dir/${data}.train.parsed

echo eval on dev data
echo python eval.py $data_dir/${data}.dev $data_dir/${data}.dev.parsed
python eval.py $data_dir/${data}.dev $data_dir/${data}.dev.parsed

echo eval on training data 
echo python eval.py $data_dir/${data}.train $data_dir/${data}.train.parsed
python eval.py $data_dir/${data}.train $data_dir/${data}.train.parsed
