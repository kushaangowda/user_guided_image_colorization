# User_Guided_Image_Colorization

export LD_LIBRARY_PATH=""

torchrun --nproc_per_node=1 main.py --batch_size=16 --epochs=20 --data_path="image_data_1k.h5" --mode="train" --type=1

torchrun --nproc_per_node=1 main.py --batch_size=1 --data_path="image_data_1k.h5" --mode="predict" --type=1 --path=""
