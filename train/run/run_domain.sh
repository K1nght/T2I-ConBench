# Define all scripts that need to be run
scripts=(
    "domain/domain_seqft.sh"
    "domain/domain_hft.sh"
    "domain/domain_mofo.sh"
    "domain/domain_replay.sh"
    "domain/domain_ewc.sh"
    "domain/domain_l2norm.sh"
    "domain/domain_joint.sh"
    "domain/domain_seqlora.sh"
    "domain/domain_inclora.sh"
    "domain/domain_clora.sh"
    "domain/domain_olora.sh"
)

# Loop through and execute each script
for script in "${scripts[@]}"; do
    bash "run/$script"
done