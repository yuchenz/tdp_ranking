
data_dir=../all_annotated_data

data=$1
model=baseline
labeled=$2
default_label=$3

echo $data $model 

if [ -f $data_dir/${data}.dev.${model}.parsed ]; 
then
    echo mv $data_dir/${data}.dev.${model}.parsed ~/.recycle
    mv $data_dir/${data}.dev.${model}.parsed ~/.recycle
fi

if [ -f $data_dir/${data}.train.${model}.parsed ]; 
then
    echo mv $data_dir/${data}.train.${model}.parsed ~/.recycle
    mv $data_dir/${data}.train.${model}.parsed ~/.recycle
fi

echo python parse.py $data_dir/${data}.dev baseline baseline $data_dir/${data}.dev.${model}.parsed baseline $labeled $default_label
python parse.py $data_dir/${data}.dev baseline baseline $data_dir/${data}.dev.${model}.parsed baseline $labeled $default_label
echo python parse.py $data_dir/${data}.train baseline baseline $data_dir/${data}.train.${model}.parsed baseline $labeled $default_label
python parse.py $data_dir/${data}.train baseline baseline $data_dir/${data}.train.${model}.parsed baseline $labeled $default_label

echo python eval.py $data_dir/${data}.dev $data_dir/${data}.dev.${model}.parsed $labeled
python eval.py $data_dir/${data}.dev $data_dir/${data}.dev.${model}.parsed $labeled

echo python eval.py $data_dir/${data}.train $data_dir/${data}.train.${model}.parsed $labeled
python eval.py $data_dir/${data}.train $data_dir/${data}.train.${model}.parsed $labeled
