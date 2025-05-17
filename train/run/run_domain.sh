# 定义所有需要运行的脚本
scripts=(
    "domain/domain_seqft.sh"
    "domain/domain_hft.sh"
    "domain/domain_mofo.sh"
    "domain/domain_replay.sh"
    "domain/domain_ewc.sh"
    "domain/domain_l2norm.sh"
    "domain/domain_joint.sh"
)

# 循环执行每个脚本
for script in "${scripts[@]}"; do
    bash "run/$script"
done