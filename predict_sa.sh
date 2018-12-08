export BERT_BASE_DIR=/data/notebooks/xff/bert/bert-master/chinese_L-12_H-768_A-12
export SA_DIR=/data/notebooks/xff/bert/sa
export TRAINED_CLASSIFIER=/data/notebooks/xff/bert/output/sa_output/model.ckpt-617


/data/miniconda3/envs/tf1.11.0/bin/python run_classifier.py \
  --task_name=sa \
  --do_predict=true \
  --data_dir=$SA_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER \
  --max_seq_length=128 \
  --output_dir=/data/notebooks/xff/bert/output/sa_output/
                           
                            
                            
                    
