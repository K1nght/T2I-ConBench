# Define all scripts that need to be run
scripts=(
    "inference/domain.sh"
    "inference/item.sh"
    "inference/comp.sh"
    "inference/cross.sh"
    "inference/fid.sh"
)

# Loop through and execute each script
for script in "${scripts[@]}"; do
    bash "run/$script"
done