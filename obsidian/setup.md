git clone https://github.com/brandf/GeodesicLatentTrajectories.git
cd GeodesicLatentTrajectories
curl -LsSf https://astral.sh/uv/install.sh | sh
~/.local/bin/uv venv
uv sync --extra gpu
source .venv/bin/activate
wandb login
WANDB_RUN=<RUN_NAME> bash run10.sh --gpu 5090|h100 --glt --viz_enabled=True --viz_scalar_every=20 --viz_hist_every=200