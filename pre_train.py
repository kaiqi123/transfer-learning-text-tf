import tensorflow as tf
import argparse
import os
from auto_encoder import AutoEncoder
from data_utils import build_word_dict, build_word_dataset, batch_iter, download_dbpedia
import time

BATCH_SIZE = 32
NUM_EPOCHS = 200
MAX_DOCUMENT_LEN = 100
num_train = 5816

def train(train_x, train_y, word_dict, args):
    with tf.Session() as sess:
        if args.model == "auto_encoder":
            model = AutoEncoder(word_dict, MAX_DOCUMENT_LEN)
        else:
            raise ValueError("Not found model: {}.".format(args.model))

        # Define training procedure
        global_step = tf.Variable(0, trainable=False)
        params = tf.trainable_variables()
        gradients = tf.gradients(model.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        optimizer = tf.train.AdamOptimizer(0.001)
        train_op = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)

        # Summary
        # loss_summary = tf.summary.scalar("loss", model.loss)
        # summary_op = tf.summary.merge_all()
        # summary_writer = tf.summary.FileWriter(args.save, sess.graph)

        # Checkpoint
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(batch_x):
            feed_dict = {model.x: batch_x}
            _, step, loss = sess.run([train_op, global_step, model.loss], feed_dict=feed_dict)
            # summary_writer.add_summary(summaries, step)

            if step % 100 == 0:
                print("step {0} : loss = {1}".format(step, loss))
                with open("pre-train-loss-all-"+args.save+".txt", "a") as f:
                    print("step {0} : loss = {1}".format(step, loss), file=f)

        # Training loop
        batches = batch_iter(train_x, train_y, BATCH_SIZE, NUM_EPOCHS)

        st = time.time()
        for batch_x, _ in batches:
            train_step(batch_x)
            step = tf.train.global_step(sess, global_step)

            steps_per_epoch = int(num_train/BATCH_SIZE)
            if step % steps_per_epoch == 0:
                print("epoch: {}, step: {}, steps_per_epoch: {}".format(int(step/steps_per_epoch), step, steps_per_epoch))
                saver.save(sess, os.path.join(args.save, "model", "model.ckpt"), global_step=step)
                print("save to {}, time of one epoch: {}".format(args.save, time.time()-st))
                st = time.time()


if __name__ == "__main__":
    stt = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="auto_encoder", help="auto_encoder | language_model")
    parser.add_argument("--save", type=str, default="save_model_auto_encoder_all_delete_5000_domainword")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    # if not os.path.exists("dbpedia_csv"):
    #     print("Downloading dbpedia dataset...")
    #     download_dbpedia()

    print("\nBuilding dictionary..")
    word_dict = build_word_dict()
    print("Preprocessing dataset..")
    train_x, train_y = build_word_dataset("train", word_dict, MAX_DOCUMENT_LEN)
    print("length of train_x: {}, length of train_y: {}".format(len(train_x), len(train_y)))
    train(train_x, train_y, word_dict, args)

    print("total time: {}".format(time.time() - stt))

