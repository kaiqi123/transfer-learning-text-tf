import tensorflow as tf
import argparse
import os
from word_rnn import WordRNN
from data_utils import build_word_dict, build_word_dataset, batch_iter, download_dbpedia
import time

NUM_CLASS = 2
BATCH_SIZE = 256
NUM_EPOCHS = 25
MAX_DOCUMENT_LEN = 100
num_train = 5816
num_test = 415

def train(train_x, train_y, test_x, test_y, vocabulary_size, args):
    with tf.Session() as sess:
        model = WordRNN(vocabulary_size, MAX_DOCUMENT_LEN, NUM_CLASS) # vocabulary_size: 268970

        # Define training procedure
        global_step = tf.Variable(0, trainable=False)
        params = tf.trainable_variables()
        gradients = tf.gradients(model.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        optimizer = tf.train.AdamOptimizer(model.lr)
        train_op = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)

        # optimizer = tf.train.AdamOptimizer(model.lr)
        # train_op = optimizer.apply_gradients(zip(gradients, params), global_step=global_step)

        # Summary
        loss_summary = tf.summary.scalar("loss", model.loss)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(args.summary_dir, sess.graph)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Load variables from pre-trained model
        if not args.pre_trained == "none":
            pre_trained_variables = [v for v in tf.global_variables()
                                     if (v.name.startswith("embedding") or v.name.startswith("birnn")) and "Adam" not in v.name]
            saver = tf.train.Saver(pre_trained_variables)
            ckpt = tf.train.get_checkpoint_state(os.path.join(args.restore_path, "model"))
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("11111111111--restore weights from {}".format(ckpt.model_checkpoint_path))

        def train_step(batch_x, batch_y):
            feed_dict = {
                model.x: batch_x,
                model.y: batch_y,
                model.keep_prob: 0.8,
            }
            _, step, summaries, loss = sess.run([train_op, global_step, summary_op, model.loss], feed_dict=feed_dict)
            summary_writer.add_summary(summaries, step)
            return loss

        def test_accuracy(test_x, test_y):
            test_batches = batch_iter(test_x, test_y, BATCH_SIZE, 1)
            sum_accuracy, cnt = 0., 0
            for test_batch_x, test_batch_y in test_batches:
                accuracy = sess.run(model.accuracy, feed_dict={model.x: test_batch_x, model.y: test_batch_y, model.keep_prob: 1.0})
                sum_accuracy += accuracy
                cnt += 1
            return sum_accuracy / cnt

        # Training loop
        batches = batch_iter(train_x, train_y, BATCH_SIZE, NUM_EPOCHS)

        st = time.time()
        steps_per_epoch = int(num_train / BATCH_SIZE)
        for batch_x, batch_y in batches:

            step = tf.train.global_step(sess, global_step)
            num_epoch = int(step / steps_per_epoch)
            curr_lr = sess.run(model.lr)

            # def get_lr(curr_lr, init_lr):
            #     import numpy as np
            #     t_total = NUM_EPOCHS * steps_per_epoch
            #     curr_lr = float(curr_lr)
            #     lr = 0.5 * init_lr * (1 + np.cos(np.pi * curr_lr / t_total))
            #     return lr
            # model.lr.load(get_lr(curr_lr, 0.01), session=sess)

            # if step == 10:
            #     model.lr.load(0.01, session=sess)
            #
            # if step == 20:
            #     model.lr.load(0.001, session=sess)

            loss = train_step(batch_x, batch_y)

            # if step % 1 == 0:
            if step % 10 == 0:
                test_acc = test_accuracy(test_x, test_y)
                train_acc = test_accuracy(train_x, train_y)

                mode = "w" if step == 0 else "a"
                with open(args.summary_dir + "-accuracy.txt", mode) as f:
                    print("{},{},{},{},{}".format(num_epoch, step, test_acc, train_acc, loss), file=f)

                print("epoch: {}, step: {}, loss: {}, steps_per_epoch: {}, batch size: {}".
                      format(num_epoch, step, loss, steps_per_epoch, BATCH_SIZE))
                print("test_accuracy: {}, train_accuracy: {}, learning rate: {}".format(test_acc, train_acc, curr_lr))
                print("time of one epoch: {}\n".format(time.time()-st))
                st = time.time()


if __name__ == "__main__":
    stt = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre_trained", type=str, default="auto_encoder", help="none | auto_encoder | language_model")
    parser.add_argument("--summary_dir", type=str, default="summary_classifier", help="summary dir.")
    parser.add_argument("--restore_path", type=str, default="save_model_auto_encoder")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # if not os.path.exists("dbpedia_csv"):
    #     print("Downloading dbpedia dataset...")
    #     download_dbpedia()

    print("\nBuilding dictionary..")
    word_dict = build_word_dict()
    print("Preprocessing dataset..")
    train_x, train_y = build_word_dataset("train", word_dict, MAX_DOCUMENT_LEN)
    test_x, test_y = build_word_dataset("test", word_dict, MAX_DOCUMENT_LEN)
    assert len(train_x) == len(train_y)
    assert len(test_x) == len(test_y)
    print("length of train_x: {}, length of test_x: {}".format(len(train_x), len(test_x)))
    print("length of word_dict: {}".format(len(word_dict)))
    train(train_x, train_y, test_x, test_y, len(word_dict), args)

    print("total time: {}".format(time.time() - stt))
