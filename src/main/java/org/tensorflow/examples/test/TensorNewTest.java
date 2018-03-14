package org.tensorflow.examples.test;

import org.tensorflow.*;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

public class TensorNewTest {
    private static byte[] readAllBytesOrExit(Path path) {
        try {
            return Files.readAllBytes(path);
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            System.exit(1);
        }
        return null;
    }

    public static float[] execute2(Tensor<Integer> input, Tensor<Float> dropout, Tensor<Float> labels, Tensor phase) {

        SavedModelBundle savedModelBundle = SavedModelBundle.load("model/comment_2","serving");
//        byte[] graphDef = readAllBytesOrExit(Paths.get("model/comment_2", "saved_model.pb"));
        Graph g = savedModelBundle.graph();
//        g.importGraphDef(graphDef);
//        Session session = new Session(g);
        Session session = savedModelBundle.session();
//        Iterator<Operation> op = g.operations();
//        while (op.hasNext()) {
//            Operation t = op.next();
//            System.out.println("name = " + t.name());
//        }
        List<Tensor<?>> result1 = session.runner()
                .feed("input_train_feature", input)
                .feed("input_dropout", dropout)
                .feed("input_train_label", labels)
                .feed("input_phase", phase)
                .fetch("output").run();
        Tensor<Float> result = result1.get(0).expect(Float.class);
        final long[] rshape = result.shape();
        if (result.numDimensions() != 2 || rshape[0] != 1) {
            throw new RuntimeException(
                    String.format(
                            "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
                            Arrays.toString(rshape)));
        }
        int nlabels = (int) rshape[1];
        input.close();
        dropout.close();
        labels.close();
        phase.close();
//        System.out.println("wait");
//        session.close();


        float[] result_i = result.copyTo(new float[1][nlabels])[0];
//        g.close();
//        try {
//            System.out.println("sleep 5s");
//            Thread.sleep(15 * 1000);
//        } catch (InterruptedException e) {
//            e.printStackTrace();
//        }
        savedModelBundle.close();
//        System.out.println("wait");
        return result_i;
    }

    public static void main(String[] args) {
        String path = args[0];
//        String path = "data/2018-03-05_03-15";
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new FileReader(path));
            String line = null;
            int index = 0;
            while ((line = reader.readLine()) != null) {
                String[] elems = line.trim().split(",");
                int label = Integer.parseInt(elems[0]);
                String uid = elems[1];
                String newsid1 = elems[2];
                String newsid2 = elems[3];
                List<Integer> features = new ArrayList<>();
                if (elems.length - 4 != 50) {
                    System.out.println(elems.length);
                    continue;
                }
                for (int i = 0; i < 50; i++) {
                    features.add(Integer.valueOf(elems[i + 4]));
                }
                if (features.size() != 50) {
                    continue;
                }
                int[] features_array = new int[50];
                for (int j = 0; j < 50; j++) {
                    features_array[j] = features.get(j).intValue();
                }
                Boolean input_phase = false;
//                int input_phase = 1;
                long[] floats_shape = {1, 50};
                long[] shape2 = {2};
                long[] shape3 = {1, 1};
                float[] input_train_label = {(float) 0};
                float[] input_dropout = {(float) 1.0, (float) 1.0};
                Tensor<Integer> features_tf = Tensor.create(floats_shape, IntBuffer.wrap(features_array));
                Tensor<Float> input_dropout_tf = Tensor.create(shape2, FloatBuffer.wrap(input_dropout));
                Tensor<Float> input_train_label_tf = Tensor.create(shape3, FloatBuffer.wrap(input_train_label));
                Tensor input_phase_tf = Tensor.create(input_phase);
                float[] res = execute2(features_tf, input_dropout_tf, input_train_label_tf, input_phase_tf);
                features_tf.close();
                input_dropout_tf.close();
                input_train_label_tf.close();
                input_phase_tf.close();
                if (index % 100 == 99) {
                    System.out.println(res[0]);
                }
                index += 1;
            }
        } catch (java.io.IOException e) {
            e.printStackTrace();
        }
    }
}
