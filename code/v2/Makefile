# On cloud.
JOBNAME=rl_breakout_$(date -u +%y%m%d_%H%M%S)
REGION= REGION
Example('us-central1')
BUCKET=SAMPLE_BUCKET_NAME
MODEL=SAMPLE_MODEL_PATH

run-local:
    gcloud ml-engine local train \
       --module-name=trainer.trainer \
       --package-path=${PWD}/rl_on_gcp/trainer \
       --\
       --steps=500000\
       --start_train=5000\
       --buffer_size=10000\
       --save_model=True\
       --model_dir='my_model'


run-cloud:
    gcloud ml-engine jobs submit training $JOBNAME \
        --package-path=$PWD/rl_on_gcp/trainer \
        --module-name=trainer.trainer \
        --region=$REGION \
        --staging-bucket=gs://$BUCKET \
        --scale-tier=BASIC_GPU\
        --runtime-version=1.10 \
        --\
        --steps=5000000\
        --start_train=5000\
        --buffer_size=500000\
        --save_model=True\
        --model_dir='gs://BUCKET/MODEL'