test_txt_dir=/home/pedroosorio/data-lefusion/Pathological/test.txt
dataset_root_dir=/home/pedroosorio/data-lefusion/Normal/Image/
train_num_steps=50001
python train/train.py dataset.test_txt_dir=$test_txt_dir dataset.root_dir=$dataset_root_dir model.train_num_steps=$train_num_steps
