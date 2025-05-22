# Define all scripts that need to be run
scripts=(
    "item-domain/item-domain_seqft.sh"
    "item-domain/item-domain_hft.sh"
    "item-domain/item-domain_mofo.sh"
    "item-domain/item-domain_replay.sh"
    "item-domain/item-domain_ewc.sh"
    "item-domain/item-domain_l2norm.sh"
    "item-domain/item-domain_joint.sh"
    "item-domain/item-domain_seqlora.sh"
    "item-domain/item-domain_inclora.sh"
    "item-domain/item-domain_clora.sh"
    "item-domain/item-domain_olora.sh"
)

# Loop through and execute each script
for script in "${scripts[@]}"; do
    bash "run/$script"
done