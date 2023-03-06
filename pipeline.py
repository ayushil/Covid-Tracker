import kfp
import yaml
from kfp import dsl
from typing import NamedTuple

from mlp_components.common import components, transformers
from mlp_components.sagemaker import trainingjob_op
from mlp_components.spark import sparkapplication_op


# Platform components
status_op = components.load('slack/status', version='1.1')

# Custom components
seed_op = kfp.components.load_component('components/seed/seed.yaml')

# Spark job
with open('components/pre-processing/pre-processing.yaml', 'r') as f:
    preprocessing_manifest = yaml.safe_load(f)

# Training Job
with open('components/train/train.yaml', 'r') as f:
    training_job = yaml.safe_load(f)

# Adhoc component
@components.adhoc()
def config_op() -> NamedTuple('output', [('bucket', str), ('kms_key_id', str)]):
    import os
    env = os.environ.get('ENV')

    if env == 'prd':
        return ('mlp-prd-4903311687432659087-724122968861', 'arn:aws:kms:us-west-2:724122968861:key/b20c2cd2-5f46-4fef-b937-f99cae01c870',)
    if env == 'pci':
        return ('mlp-pci-4903311687432659087-724122968861', 'arn:aws:kms:us-west-2:724122968861:key/059607f2-a49d-4008-a764-65060d06caf0',)
    if env == 'e2e':
        return ('mlp-e2e-4903311687432659087-435945521637', 'arn:aws:kms:us-west-2:435945521637:key/01a040ac-07a3-4132-b0d2-6c747439f966',)

    raise Exception(f'Unknown environment {env}')


# Pipeline
@dsl.pipeline()
def pipeline():
    dsl.get_pipeline_conf().add_op_transformer(transformers.mlp_transformer())

    slack_channel = '#mlp-upskill'  # replace with your own channel

    # Send slack message when pipeline finishes
    notify_finished = status_op(
        name='{{workflow.name}}',
        status='{{workflow.status}}',
        channel=slack_channel)

    with dsl.ExitHandler(notify_finished):
        # Send slack message when pipeline starts
        notify_started = status_op(
            name='{{workflow.name}}',
            status='Started',
            channel=slack_channel)

        config = config_op().after(notify_started)

        params = {
            'bucket': config.outputs['bucket'],
            'kms_key_id': config.outputs['kms_key_id'],
            'filename': '{{workflow.name}}/data.txt',
            'processed_data': '{{workflow.name}}/processed_data'
        }

        seed = seed_op(config.outputs['bucket'], params['filename'])

        for index, value in enumerate(preprocessing_manifest["spec"]["arguments"]):
            preprocessing_manifest["spec"]["arguments"][index] = value.format(**params)

        pre_processing = sparkapplication_op(manifest=preprocessing_manifest).after(seed)

        training_job['spec']['inputDataConfig'][0]['dataSource']['s3DataSource']['s3URI'] \
            = training_job['spec']['inputDataConfig'][0]['dataSource']['s3DataSource']['s3URI'].format(**params)

        training_job['spec']['outputDataConfig']['s3OutputPath'] \
            = training_job['spec']['outputDataConfig']['s3OutputPath'].format(**params)

        training_job['spec']['outputDataConfig']['kmsKeyID'] \
            = training_job['spec']['outputDataConfig']['kmsKeyID'].format(**params)

        trainingjob_op(manifest=training_job).after(pre_processing)
