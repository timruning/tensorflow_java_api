package org.tensorflow.examples.comment;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.nio.IntBuffer;
import java.util.Arrays;
import java.util.List;

public class MultiThread implements Runnable {
    private Thread t;
    private String threadName;
    private List<List<Integer>> tmp_sample;
    //    private Graph g;
    private Session s;

    MultiThread(String name, Session s, List<List<Integer>> data) {
        this.threadName = name;
//        this.g = g;
        this.s = s;
        this.tmp_sample = data;
    }

    public void start() {
        System.out.println("starting " + threadName);
        if (t == null) {
            t = new Thread(this, threadName);
            t.start();
        }
    }

    private float[][] execute(Tensor<Integer> input, Tensor<Float> dropout, Tensor<Float> labels, Tensor phase, int num) {

        List<Tensor<?>> result1 = s.runner()
                .feed("input_train_feature", input)
                .feed("input_dropout", dropout)
                .feed("input_train_label", labels)
                .feed("input_phase", phase)
                .fetch("output")
                .run();
        Tensor<?> result = result1.get(0);
        final long[] rshape = result.shape();
        if (result.numDimensions() != 2 || rshape[0] != num) {
            throw new RuntimeException(
                    String.format(
                            "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
                            Arrays.toString(rshape)));
        }
        int batch = (int) rshape[0];
        int nlabels = (int) rshape[1];
        return result.copyTo(new float[batch][nlabels]);
    }

    @Override
    public void run() {
        int index = 0;
        long time = System.currentTimeMillis();
        int batch_size = 10;
        int label_size = 50;
        short[][] features = new short[batch_size][label_size];
        float[][] labels = new float[batch_size][1];
        float[][] input_dropout = new float[batch_size][2];
        Boolean input_phase = false;
        long[] floats_shape = {batch_size, label_size};
        long[] shape2 = {batch_size, 2};
        long[] shape3 = {batch_size, 1};
        float[] inpt_dropout2 = {(float) 1.0, (float) 1.0};
        int[] features2 = new int[batch_size * label_size];
        while (index < tmp_sample.size()) {
            List<Integer> elem = tmp_sample.get(index);

            int a = index % batch_size;

            labels[a][0] = 0;
            input_dropout[a][0] = (float) 1.0;
            input_dropout[a][1] = (float) 1.0;
            for (int j = 0; j < 50; j++) {
                features[a][j] = (short) elem.get(j).intValue();
                features2[a * label_size + j] = elem.get(j).intValue();
            }
            if (a == batch_size - 1) {

                Tensor<Integer> features_tf = Tensor.create(floats_shape, IntBuffer.wrap(features2));
                int[][] x = features_tf.copyTo(new int[batch_size][label_size]);
//                System.out.println("x");
                Tensor<Float> input_dropout_tf = Tensor.create(inpt_dropout2, Float.class);
                Tensor<Float> input_train_label_tf = Tensor.create(labels, Float.class);
                Tensor input_phase_tf = Tensor.create(input_phase);
                float[][] res = execute(features_tf, input_dropout_tf, input_train_label_tf, input_phase_tf, batch_size);
//                for (int i = 0; i < batch_size; i++) {
//                    System.out.println(res[i][0]);
//                }
//                try {
//                    Thread.sleep(1000 * 10);
//                } catch (InterruptedException e) {
//                    e.printStackTrace();
//                }
            }
            if (index % 100000 == 99999) {
                long time2 = System.currentTimeMillis();
                System.out.println(time2 - time);
                time = time2;
            }
            index += 1;
        }
    }
}
