git clone https://github.com/brandf/GeodesicLatentTrajectories.git
cd GeodesicLatentTrajectories
curl -LsSf https://astral.sh/uv/install.sh | sh
~/.local/bin/uv venv
source .venv/bin/activate
uv sync --extra gpu
wandb login
[export GLT_MEM_DEBUG=1]
WANDB_RUN=<RUN_NAME> bash run10.sh --gpu 5090|h100 --glt --viz_enabled=True --viz_scalar_every=20 --viz_hist_every=200 --glt_offsets="[-1,0,1]" --glt_ce_chunk 2 [--glt_no_geom]