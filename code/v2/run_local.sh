# On cloud.
JOBNAME=rl_halite_$(date -u +%y%m%d_%H%M%S)
REGION=us-central1
# Example('us-central1')
# --billing-project=precise-ego-292802
BUCKET=zp-halite-playground
MODEL=dqn_basic_2020_01/$JOBNAME/
JOB_OUTPUT_PATH=gs://$BUCKET/$MODEL

gcloud ai-platform local train \
   --module-name=trainer.task \
   --package-path=${PWD}/rl_on_gcp/trainer \
   --\
   --steps=400\
   --buffer_size=5000\
   --save_model=True\
   --model_dir=$MODEL\