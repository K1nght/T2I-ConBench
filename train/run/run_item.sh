# 定义所有需要运行的脚本
scripts=(
    "item/item_seqft.sh"
    "item/item_hft.sh"
    "item/item_mofo.sh"
    "item/item_replay.sh"
    "item/item_ewc.sh"
    "item/item_l2norm.sh"
    "item/item_joint.sh"
)

# 循环执行每个脚本
for script in "${scripts[@]}"; do
    bash "run/$script"
done