# text classification
## scikit-learn

* see [sklearn-tutorial](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
* [minimal document classification](minimal_example/readme.md)

## huggingface transformers

1. __locally__ do rsync: `rsync -avz -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --exclude=.git --max-size=1m /home/tilo/code/NLP/transformers tilo-himmelsbach@gateway.hpc.tu-berlin.de:/home/users/t/tilo-himmelsbach/`
2. __on frontend__: `OMP_NUM_THREADS=2 wandb init`
3. __on gpu-node__: `WANDB_MODE=dryrun python3 run_pl_glue.py --data_dir ~/data/glue/MRPC --task mrpc --model_name_or_path bert-base-cased --output_dir ~/data/lightning_experiments/glue_mrpc --max_seq_length  128 --learning_rate 2e-5 --num_train_epochs 1 --train_batch_size 32 --seed 2 --do_train --do_predict --gpus 1`
4. __on frontend__ syncing to wandb: `OMP_NUM_THREADS=4 wandb sync wandb/dryrun-...`
5. see [results](https://app.wandb.ai/dertilo/text-classification/runs/1o2j6s2m/overview?workspace=user-)