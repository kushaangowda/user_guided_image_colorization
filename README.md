# User_Guided_Image_Colorization

torchrun --nproc_per_node=1 main.py --batch_size=32 --epochs=10 --data_path="image_data_1000img.h5" --mode="train"

torchrun --nproc_per_node=1 main.py --batch_size=32 --data_path="image_data_1000img.h5" --mode="predict"
