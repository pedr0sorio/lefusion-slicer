test_txt_dir=/home/pedroosorio/data-lefusion/Pathological/test.txt
dataset_root_dir=/home/pedroosorio/data-lefusion/Normal/Image/
target_img_path=/home/pedroosorio/data-lefusion/gen/Image/
target_label_path=/home/pedroosorio/data-lefusion/gen/Mask/
jump_length=5
jump_n_sample=5

python test/inference.py test_txt_dir=$test_txt_dir dataset_root_dir=$dataset_root_dir target_img_path=$target_img_path target_label_path=$target_label_path schedule_jump_params.jump_length=$jump_length schedule_jump_params.jump_n_sample=$jump_n_sample
