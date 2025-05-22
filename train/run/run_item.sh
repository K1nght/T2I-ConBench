# Define all scripts that need to be run
scripts=(
    "item/item_seqft.sh"
    "item/item_hft.sh"
    "item/item_mofo.sh"
    "item/item_replay.sh"
    "item/item_ewc.sh"
    "item/item_l2norm.sh"
    "item/item_joint.sh"
    "item/item_seqlora.sh"
    "item/item_inclora.sh"
    "item/item_olora.sh"
    "item/item_clora.sh"
)

# Loop through and execute each script
for script in "${scripts[@]}"; do
    bash "run/$script"
done