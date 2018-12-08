from __future__ import print_function
import os.path

# Install tensorflow APIs with tensorflow-serving-api, gpu version is not a must
import tensorflow as tf
from run_classifier_predict_online import * 


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("model_output_path", "/data/notebooks/xff/bert/output/sa_output/saved_model/", "model output path")
tf.app.flags.DEFINE_string("version", "1", "model version")

def export(): 
    
    with graph.as_default():
        
        # below is already defined in run_classifier_predict_online.py
        
        #input_ids_p = tf.placeholder(tf.int32, [batch_size, FLAGS.max_seq_length], name="input_ids")
        #input_mask_p = tf.placeholder(tf.int32, [batch_size, FLAGS.max_seq_length], name="input_mask")
        #label_ids_p = tf.placeholder(tf.int32, [batch_size], name="label_ids")
        #segment_ids_p = tf.placeholder(tf.int32, [FLAGS.max_seq_length], name="segment_ids")
        #total_loss, per_example_loss, logits, probabilities,model = create_model(
        #   bert_config, is_training, input_ids_p, input_mask_p, segment_ids_p,
        #   label_ids_p, num_labels, use_one_hot_embeddings)
                
        possibility = probabilities[0]
        label_predict = tf.argmax(possibility)
                
        saver = tf.train.Saver()
        saver.restore(sess, FLAGS.init_checkpoint)               
        
        export_path = os.path.join(FLAGS.model_output_path, FLAGS.version)
        print("Exporting trained model to %s, version: %s" % (FLAGS.model_output_path, FLAGS.version))
    
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    
        input_ids_tensor_info = tf.saved_model.utils.build_tensor_info(input_ids_p)
        input_mask_tensor_info = tf.saved_model.utils.build_tensor_info(input_mask_p)
        segment_ids_tensor_info = tf.saved_model.utils.build_tensor_info(segment_ids_p)
        label_ids_tensor_info = tf.saved_model.utils.build_tensor_info(label_ids_p)
        
        
        classes_output_tensor_info = tf.saved_model.utils.build_tensor_info(label_predict)
        scores_output_tensor_info = tf.saved_model.utils.build_tensor_info(possibility)
    
        vectorize_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs = {"input_ids": input_ids_tensor_info,
                              "input_mask": input_mask_tensor_info,
                              "segment_ids": segment_ids_tensor_info,
                              "label_ids": label_ids_tensor_info
                             },
                    outputs = {
                               "label_predict": classes_output_tensor_info,
                               "possibility": scores_output_tensor_info
                    },
                    method_name="tensorflow/serving/predict"
                )
            )

        legacy_init_op = tf.group(tf.tables_initializer(), name = "legacy_init_op")

        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                                 signature_def_map={"vectorize_service": vectorize_signature},
                                                 legacy_init_op=legacy_init_op)
        
        builder.save()
        

if __name__ == "__main__":
    export()