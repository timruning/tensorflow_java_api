package org.tensorflow.examples.image;

import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

/**
 * Created by zhongnanhuang209074 on 2018/1/19.
 */
public class LabelInageT {

    private List<String> labels;
    private Graph g;
    private Session s;

    private LabelInageT(String modelpath){
        //加载模型文件
        byte[] graphDef = readAllBytesOrExit(Paths.get(modelpath, "tensorflow_inception_graph.pb"));
        g = new Graph();
        g.importGraphDef(graphDef);
        s = new Session(g);

        //加载标签文件
        labels = readAllLinesOrExit(Paths.get(modelpath, "imagenet_comp_graph_label_strings.txt"));
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
    private static List<String> readAllLinesOrExit(Path path) {
        try {
            return Files.readAllLines(path, Charset.forName("UTF-8"));
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            System.exit(0);
        }
        return null;
    }
    private Tensor<Float> constructAndExecuteGraphToNormalizeImage(byte[] imageBytes) {

        Graph imgG = new Graph();
        Session imgS = new Session(imgG);
        GraphBuilder b = new GraphBuilder(imgG);
        final int H = 224;
        final int W = 224;
        final float mean = 117f;
        final float scale = 1f;
        final Output<String> input = b.constant("input", imageBytes);
        final Output<Float> output =
                b.div(
                        b.sub(
                                b.resizeBilinear(
                                        b.expandDims(
                                                b.cast(b.decodeJpeg(input, 3), Float.class),
                                                b.constant("make_batch", 0)),
                                        b.constant("size", new int[] {H, W})),
                                b.constant("mean", mean)),
                        b.constant("scale", scale));

        return imgS.runner().fetch(output.op().name()).run().get(0).expect(Float.class);
    }

    private float[] executeInceptionGraph(Tensor<Float> image) {

        Tensor<Float> result = s.runner().feed("input", image).fetch("output").run().get(0).expect(Float.class);
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

    /**
     * 获取到最大的数
     * @param probabilities
     * @return
     */
    private static int maxIndex(float[] probabilities) {
        int best = 0;
        for (int i = 1; i < probabilities.length; ++i) {
            if (probabilities[i] > probabilities[best]) {
                best = i;
            }
        }
        return best;
    }

    public void classification(String imgpath){
        //读取要分类的图片文件
        byte[] imageBytes = readAllBytesOrExit(Paths.get(imgpath));
        Tensor<Float> image = constructAndExecuteGraphToNormalizeImage(imageBytes);
        float[] labelProbabilities = executeInceptionGraph(image);
        int bestLabelIdx = maxIndex(labelProbabilities);
//        System.out.println(
//                String.format("BEST MATCH: %s (%.2f%% likely)",
//                        labels.get(bestLabelIdx),
//                        labelProbabilities[bestLabelIdx] * 100f));
    }


    public static void main(String[] args){
        LabelInageT labelInageT = new LabelInageT("model/inception5h");

        long start = System.currentTimeMillis();
        for (int i = 0;i < 100;i++) {
            labelInageT.classification("model/test.jpg");
        }
        System.out.println("avg deal time is "+(System.currentTimeMillis()-start)/(100*1.0));
    }

}
