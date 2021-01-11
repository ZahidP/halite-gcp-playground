# On cloud.
JOBNAME=rl_halite_$(date -u +%y%m%d_%H%M%S)
REGION=us-central1
BUCKET=zp-halite-playground
MODEL=dqn_basic_2020_01/$JOBNAME/
JOB_OUTPUT_PATH=gs://$BUCKET/$MODEL

gcloud ai-platform jobs submit training $JOBNAME \
    --package-path=$PWD/rl_on_gcp/trainer \
    --module-name=trainer.task \
    --region=$REGION \
    --staging-bucket=gs://$BUCKET \
    --scale-tier=BASIC\
    --runtime-version=2.2 \
    --python-version=3.7 \
    --\
    --steps=4000\
    --buffer_size=800\
    --save_model=True\
    --model_dir=gs://$BUCKET/$MODEL