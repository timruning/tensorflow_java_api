package org.tensorflow.examples.comment;

import org.tensorflow.*;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Created by zhongnanhuang209074 on 2018/1/19.
 */
public class CommentClassification {


    private Graph g;
    private Session s;
    SavedModelBundle ss;

    public CommentClassification(String modelpath) {
        //加载模型文件
//        byte[] graphDef = readAllBytesOrExit(Paths.get("/opt/develop/workspace/sohu/NFM/tensorflow_java/model/comment_1", "graph.pb"));
//        byte[] graphDef2 = readAllBytesOrExit(Paths.get(modelpath, "graph.pb"));
//        g = new Graph();
//        g.importGraphDef(graphDef);
//
//        s = new Session(g);
        ss = SavedModelBundle.load("/opt/develop/workspace/sohu/NFM/tensorflow_java/model/comment_1");
        ss = SavedModelBundle.load(modelpath + "_2");
        ss = SavedModelBundle.load(modelpath + "_1");
        ss = SavedModelBundle.load(modelpath + "_2");
        ss = SavedModelBundle.load(modelpath + "_1");
        ss = SavedModelBundle.load(modelpath + "_2");
        ss = SavedModelBundle.load(modelpath + "_1");
        ss = SavedModelBundle.load(modelpath + "_2");
        ss = SavedModelBundle.load(modelpath + "_1");
        ss = SavedModelBundle.load(modelpath + "_2");
        ss = SavedModelBundle.load(modelpath + "_1");
        ss = SavedModelBundle.load(modelpath + "_2");
        ss = SavedModelBundle.load(modelpath + "_1");
        ss = SavedModelBundle.load(modelpath + "_2");
        ss = SavedModelBundle.load(modelpath + "_2");
        ss = SavedModelBundle.load(modelpath + "_1");
        ss = SavedModelBundle.load(modelpath + "_2");
        ss = SavedModelBundle.load(modelpath + "_1");
        ss = SavedModelBundle.load(modelpath + "_1");
        ss = SavedModelBundle.load(modelpath + "_2");
        ss = SavedModelBundle.load(modelpath + "_1");
        ss = SavedModelBundle.load(modelpath + "_2");
        ss = SavedModelBundle.load(modelpath + "_1");
        ss = SavedModelBundle.load(modelpath + "_2");
        ss = SavedModelBundle.load(modelpath + "_1");
    }

    private static byte[] readAllBytesOrExit(Path path) {
        try {
            return Files.readAllBytes(path);
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            System.exit(1);
        }
        return null;
    }


    private float[] execute2(Tensor<Integer> input, Tensor<Float> dropout, Tensor<Float> labels, Tensor phase) {

        List<Tensor<?>> result1 = s.runner()
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
        return result.copyTo(new float[1][nlabels])[0];
    }

    public void classification3(String path, int thread_num) {
//        String path = "data/2018-03-05_03-15";
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new FileReader(path));
            String line = null;
            Random random = new Random();
            List<List<Integer>> tmp_sample = new ArrayList<>();
            while ((line = reader.readLine()) != null) {
                String[] elems = line.trim().split(",");
                int label = Integer.parseInt(elems[0]);
                String uid = elems[1];
                String newsid1 = elems[2];
                String newsid2 = elems[3];
                List<Integer> feature = new ArrayList<>();
                if (elems.length - 4 != 50) {
                    System.out.println(elems.length);
                    continue;
                }
                for (int i = 0; i < 50; i++) {
                    feature.add(Integer.valueOf(elems[i + 4]));
                }
                if (feature.size() != 50) {
                    continue;
                } else {
//                    System.out.println(uid + "\t" + newsid1 + "\t" + newsid2);
                    tmp_sample.add(feature);
                }
            }

            for (int i = 0; i < thread_num; i++) {
                String thread_name = "new_threamd_" + i;
                MultiThread thread = new MultiThread(thread_name, s, tmp_sample);
                thread.start();
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

//    public void classification2() {
//        String path = "data/2018-03-05_03-15";
//        BufferedReader reader = null;
//        try {
//            reader = new BufferedReader(new FileReader(path));
//            String line = null;
//            Random random = new Random();
//            List<List<Integer>> tmp_sample = new ArrayList<>();
//            int index = 0;
//            int randNum = random.nextInt(20) + 1;
//            while (index < 3) {
//                randNum = 3;
//                if (tmp_sample.size() < randNum) {
//                    line = reader.readLine();
//                    if (line == null) {
//                        break;
//                    }
//                    String[] elems = line.trim().split(",");
//                    int label = Integer.parseInt(elems[0]);
//                    String uid = elems[1];
//                    String newsid1 = elems[2];
//                    String newsid2 = elems[3];
//                    List<Integer> feature = new ArrayList<>();
//                    if (elems.length - 4 != 50) {
//                        System.out.println(elems.length);
//                        continue;
//                    }
//                    for (int i = 0; i < 50; i++) {
//                        feature.add(Integer.valueOf(elems[i + 4]));
//                    }
//                    if (feature.size() != 50) {
//                        continue;
//                    } else {
//                        System.out.println(uid + "\t" + newsid1 + "\t" + newsid2);
//                        tmp_sample.add(feature);
//                    }
//                } else {
//                    float[] input_dropout = {(float) 0.8, (float) 0.5};
//                    Boolean input_phase = true;
//                    long[] floats_shape = {randNum, 50};
//                    long[] shape2 = {2};
//                    long[] shape3 = {randNum, 1};
//                    float[] input_train_label = new float[randNum];
//                    for (int i = 0; i < randNum; i++) {
//                        input_train_label[i] = 0;
//                    }
//                    int[][] features_final = new int[randNum][50];
//                    for (int i = 0; i < randNum; i++) {
//                        for (int j = 0; j < 50; j++) {
//                            features_final[i][j] = tmp_sample.get(i).get(j).intValue();
//                        }
//                    }
//                    Tensor<Integer> features_tf = Tensor.create(features_final, Integer.class);
//                    Tensor<Float> input_dropout_tf = Tensor.create(shape2, FloatBuffer.wrap(input_dropout));
//                    Tensor<Float> input_train_label_tf = Tensor.create(shape3, FloatBuffer.wrap(input_train_label));
//                    Tensor input_phase_tf = Tensor.create(input_phase);
//                    float[][] res = execute(features_tf, input_dropout_tf, input_train_label_tf, input_phase_tf, randNum);
//                    for (int i = 0; i < randNum; i++) {
//                        System.out.println(res[i][0]);
//                    }
//
//                    tmp_sample.clear();
//                    randNum = random.nextInt(20) + 1;
//                    index += 1;
//                }
//
//            }
//        } catch (Exception e) {
//            e.printStackTrace();
//        }
//    }

    public void classification(String path) {
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new FileReader(path));
            String line = null;
            long index = 0;
            List<List<Integer>> data = new ArrayList<>();
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
                if (features.size() == 50) {
                    data.add(features);
                }
            }
            long time1 = System.currentTimeMillis();
            for (int i = 0; i < data.size() && i < 10; i++) {
                int[] features = new int[50];
                for (int j = 0; j < 50; j++) {
                    features[j] = data.get(i).get(j).intValue();
                }
                Boolean input_phase = false;
//                int input_phase = 1;
                long[] floats_shape = {1, 50};
                long[] shape2 = {2};
                long[] shape3 = {1, 1};
                float[] input_train_label = {(float) 0};
                float[] input_dropout = {(float) 1.0, (float) 1.0};
                Tensor<Integer> features_tf = Tensor.create(floats_shape, IntBuffer.wrap(features));
                Tensor<Float> input_dropout_tf = Tensor.create(shape2, FloatBuffer.wrap(input_dropout));
                Tensor<Float> input_train_label_tf = Tensor.create(shape3, FloatBuffer.wrap(input_train_label));
                Tensor input_phase_tf = Tensor.create(input_phase);
                float[] res = execute2(features_tf, input_dropout_tf, input_train_label_tf, input_phase_tf);
                features_tf.close();
                input_dropout_tf.close();
                input_train_label_tf.close();
                input_phase_tf.close();
                System.out.println("close");
                s.close();
                g.close();
//                System.out.println(res[0]);
                if (i % 100000 == 99999) {
                    long time2 = System.currentTimeMillis();
                    System.out.println(time2 - time1);
                    time1 = time2;
                }
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        String path = "data/2018-03-05_03-15";
//        int thread_num = 2;
//        String path = args[0];
//        int thread_num = Integer.parseInt(args[1]);

        CommentClassification commentClassification = new CommentClassification("model/comment");
//        commentClassification.classification("if you sometimes like to go to the movies to have fun , wasabi is a good place to start . ");
//        commentClassification.classification3(path, thread_num);
        commentClassification.classification(path);
    }
}