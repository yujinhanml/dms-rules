#!/bin/bash

# 基础路径
base_path="/cpfs01/user/hanyujin/causal-dm/results/sunshadow_lfd_lnd_rfd_rnd"
result_path="${base_path}/vis"

# 循环从1400到4000，步长为200
for i in $(seq 1400 200 4000)
do
    # 检查是否已经存在对应的生成文件夹
    if ls ${result_path}/epoch_${i}* 1> /dev/null 2>&1; then
        echo "Skipping checkpoint $i as generated files already exist"
        continue
    fi

    # 构建权重文件路径
    weight_path="${base_path}/ckpt_${i}.pt"
    
    # 检查权重文件是否存在
    if [ ! -f "$weight_path" ]; then
        echo "Weight file not found for checkpoint $i, skipping"
        continue
    fi

    # 运行Python命令
    python generate.py \
        --generate_name "sunshadowweight_${i}" \
        --num_images 3000 \
        --image_size 32 \
        --weight_path "$weight_path" \
        --sample ddpm \
        --result_path "$result_path" \
        --use_gpu 1
    
    echo "Completed generation for checkpoint $i"
done

echo "All generations completed"