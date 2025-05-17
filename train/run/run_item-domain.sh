# 定义所有需要运行的脚本
scripts=(
    "item-domain/item-domain_seqft.sh"
    "item-domain/item-domain_hft.sh"
    "item-domain/item-domain_mofo.sh"
    "item-domain/item-domain_replay.sh"
    "item-domain/item-domain_ewc.sh"
    "item-domain/item-domain_l2norm.sh"
    "item-domain/item-domain_joint.sh"
)

# 循环执行每个脚本
for script in "${scripts[@]}"; do
    bash "run/$script"
done