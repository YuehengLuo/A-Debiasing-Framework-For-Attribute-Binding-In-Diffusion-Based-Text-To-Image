out_path="../images/"
model="sdxl"

data_list=("example.txt")

for data in "${data_list[@]}"; do
    CUDA_VISIBLE_DEVICES=1 python inference_xl.py \
        --test_file "../test/$data" \
        --out_path "$out_path" \
        --model "$model"
done
