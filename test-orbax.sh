#!/usr/bin/env bash

set -xe

export NUM_REPLICAS=${NUM_REPLICAS:-2}
export JOBSET_NAME=${JOBSET_NAME:-$USER-regular-ckpt-gcs-500mb-pytree}
export BASTION_TIER=disabled
export GKE_CLUSTER=$(axlearn gcp config | grep gke_cluster | awk '{ print $3 }' | tr -d '"')
# Switch to tpu-v6e-256 if on scale cluster
export INSTANCE_TYPE=${INSTANCE_TYPE:-"tpu-v6e-256"}
# Switch to tpu-v6e-256-4 if on scale cluster
export MESH_SELECTOR=${MESH:-"tpu-v6e-256-4"}
export CONFIG=${CONFIG:-"fuji-70B-v2-flash-orbax"}
export PROJECT_ID=$(gcloud config get project)
export ENABLE_GCSFUSE=${ENABLE_GCSFUSE:-true}

# Example for v6e-256
# MESH_SELECTOR=tpu-v6e-256-4 INSTANCE_TYPE=tpu-v6e-256 ./test-orbax.sh

# The bundle step is needed if you run on cloudtop
# uncomment if you use cloudtop
axlearn gcp bundle --name=$JOBSET_NAME \
         --bundler_spec=allow_dirty=True \
         --bundler_type=artifactregistry \
         --bundler_spec=dockerfile=Dockerfile \
         --bundler_spec=image=tpu \
         --bundler_spec=target=tpu

# Only enable kueue when running on scale testing cluster
# --queue=multislice-queue \
# --priority_class=very-high \
# --trainer_dir=gs://tess-checkpoints-us-west1/${JOBSET_NAME}-nr-${NUM_REPLICAS}/ \
#

if [[ "$CONFIG" == *"orbaxem"* && "$ENABLE_GCSFUSE" == "true" ]]; then
  echo "Running with Orbax emergency checkpointer and GCSFuse enabled."
  axlearn gcp launch run --cluster=$GKE_CLUSTER \
       --runner_name gke_tpu_single \
       --name=$JOBSET_NAME \
       --instance_type=${INSTANCE_TYPE} \
       --priority_class=high \
       --host_mount_spec=name=tmp,host_path=/tmp,mount_path=/host-tmp \
       --gcsfuse_mount_spec=gcs_path=gs://tess-dataset-southamerica-west1 \
       --num_replicas=${NUM_REPLICAS} \
       --bundler_spec=allow_dirty=True \
       --bundler_type=artifactregistry --bundler_spec=image=tpu \
       --bundler_spec=dockerfile=Dockerfile --bundler_spec=target=tpu \
       -- "python3 -c 'import jax; jax.devices()'; python3 -m axlearn.common.launch_trainer_main" \
         --init_module=axlearn.common.checkpointer_orbax_emergency:local_ckpt_dir=/host-tmp/checkpoints \
         --module=text.gpt.c4_trainer \
         --config=${CONFIG} \
         --trainer_dir=/tmp/dataset/ \
         --data_dir=/tmp/dataset  \
         --jax_backend=tpu \
         --mesh_selector=${MESH_SELECTOR} \
         --initialization_timeout=1200 \
         --trace_at_steps=29,59,89,119,149,179,209,239,269,299.329,359,389,419,449,479,509,539,569,599,629,659,689,719
# Check if CONFIG ends with "orbaxem"
elif [[ "$CONFIG" == *"orbaxem"* ]]; then
  echo "Running with Orbax emergency checkpointer."
  echo "Running with Orbax emergency checkpointer."
 axlearn gcp launch run --cluster=$GKE_CLUSTER \
       --runner_name gke_tpu_single \
       --name=$JOBSET_NAME \
       --instance_type=${INSTANCE_TYPE} \
       --priority_class=high \
       --host_mount_spec=name=tmp,host_path=/tmp,mount_path=/host-tmp \
       --num_replicas=${NUM_REPLICAS} \
       --bundler_spec=allow_dirty=True \
       --bundler_type=artifactregistry --bundler_spec=image=tpu \
       --bundler_spec=dockerfile=Dockerfile --bundler_spec=target=tpu \
       -- "python3 -c 'import jax; jax.devices()'; python3 -m axlearn.common.launch_trainer_main" \
         --init_module=axlearn.common.checkpointer_orbax_emergency:local_ckpt_dir=/host-tmp/checkpoints \
         --module=text.gpt.c4_trainer \
         --config=${CONFIG} \
         --trainer_dir=gs://tess-dataset-southamerica-west1/deepikarajani-directtogcs-64/ \
         --data_dir=gs://tess-dataset-southamerica-west1  \
         --jax_backend=tpu \
         --mesh_selector=${MESH_SELECTOR} \
         --initialization_timeout=1200 \
         --trace_at_steps=29,59,89,119,149,179,209,239,269,299,329,359,389,419,449,479,509,539,569,599,629,659,689,719
elif [[ "$ENABLE_GCSFUSE" == "true" ]]; then
  echo "Running Orbax regular checkpointer or AXLearn native with GCSFuse enabled."
  axlearn gcp launch run --cluster=$GKE_CLUSTER \
        --runner_name gke_tpu_single \
        --name=$JOBSET_NAME \
        --instance_type=${INSTANCE_TYPE} \
        --priority_class=high \
        --num_replicas=${NUM_REPLICAS} \
        --bundler_spec=allow_dirty=True \
        --gcsfuse_mount_spec=gcs_path=gs://tess-dataset-southamerica-west1 \
        --bundler_type=artifactregistry --bundler_spec=image=tpu \
        --bundler_spec=dockerfile=Dockerfile --bundler_spec=target=tpu \
        -- "ulimit -n 1048576; ulimit -c 0; python3 -c 'import jax; jax.devices()'; python3 -m axlearn.common.launch_trainer_main" \
          --module=text.gpt.c4_trainer \
          --config=${CONFIG} \
          --trainer_dir=/tmp/dataset/regularckpt-deepikarajani-fuse-64-70b-500mb-corrected/ \
          --data_dir=/tmp/dataset \
          --jax_backend=tpu \
          --mesh_selector=${MESH_SELECTOR} \
          --initialization_timeout=1200 \
          --trace_at_steps=29,59,89,119,149,179,209,239,269,299,329,359,389,419,449,479,509,539,569,599,629,659,689,719
else
  echo "Running Orbax regular checkpointer or AXLearn native."
  axlearn gcp launch run --cluster=$GKE_CLUSTER \
        --runner_name gke_tpu_single \
        --name=$JOBSET_NAME \
        --instance_type=${INSTANCE_TYPE} \
        --priority_class=high \
        --num_replicas=${NUM_REPLICAS} \
        --bundler_spec=allow_dirty=True \
        --bundler_type=artifactregistry --bundler_spec=image=tpu \
        --bundler_spec=dockerfile=Dockerfile --bundler_spec=target=tpu \
        -- "ulimit -n 1048576; ulimit -c 0; python3 -c 'import jax; jax.devices()'; python3 -m axlearn.common.launch_trainer_main" \
          --module=text.gpt.c4_trainer \
          --config=${CONFIG} \
          --trainer_dir=gs://tess-dataset-southamerica-west1/regularckpt-deepikarajani-directtogcs-128-70b/ \
          --data_dir=gs://tess-dataset-southamerica-west1  \
          --jax_backend=tpu \
          --mesh_selector=${MESH_SELECTOR} \
          --initialization_timeout=1200 \
          --trace_at_steps=29,59,89,119,149,179,209,239,269,299,329,359,389,419,449,479,509,539,569,599,629,659,689,719
fi
