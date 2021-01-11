# HP job on cloud.
REGION=us-central1
JOBNAME=rl_breakout_hp_$(date -u +%y%m%d_%H%M%S)
BUCKET=zp-halite-playground
MODEL=dqn_basic_2020_01/$JOBNAME/

gcloud ai-platform jobs submit training $JOBNAME \
        --package-path=$PWD/rl_on_gcp/trainer \
        --module-name=trainer.task \
        --region=$REGION \
        --staging-bucket=gs://$BUCKET \
        --config=hyperparameters.yaml \
        --runtime-version=1.10 \
        --\
        --steps=4000\
        --buffer_size=1200\
        --save_model=True\
        --model_dir=gs://$BUCKET/$MODEL