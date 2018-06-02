import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.log4j.LogManager;
import org.apache.spark.SparkConf;

import java.util.HashMap;
import java.util.Map;

import org.apache.spark.mllib.tree.GradientBoostedTrees;
import org.apache.spark.mllib.tree.configuration.BoostingStrategy;
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel;
import scala.Tuple2;

public class TitanicProblemSpark {

    final static int k = 3;
    public static final String COMMA_DELIMITER = ",(?=([^\"]*\"[^\"]*\")*[^\"]*$)";

    public static void createFolds(JavaRDD<String> data) {
        for (int num = k; num > 0; num--) {
            if (num == 1) {
                System.out.println(data.count());
                data.saveAsTextFile(Integer.toString(num));
                break;
            }
            double rate = 1 / (double)num;
            JavaRDD<String>[] tmp = data.randomSplit(new double[]{rate, (1-rate)});
            JavaRDD<String> savedData = tmp[0];
            System.out.println(savedData.count());
            savedData.saveAsTextFile(Integer.toString(num));
            data = tmp[1];
        }
    }

    public static void main(String[] args) {
        LogManager.getLogger("org").setLevel(org.apache.log4j.Level.OFF);
        SparkConf sparkConf = new SparkConf().setMaster("local[*]").setAppName("Titanic Spark");
        JavaSparkContext javaSparkContext = new JavaSparkContext(sparkConf);

        JavaRDD<String> datatrain = javaSparkContext.textFile("titanic_train.csv").filter(line -> !"".equals(line.split(COMMA_DELIMITER)[5]));
        JavaRDD<String> datatest = javaSparkContext.textFile("titanic_test.csv").filter(line -> !"".equals(line.split(COMMA_DELIMITER)[5]));

        //JavaRDD<String>[] tmp = data.randomSplit(new double[]{0.1, 0.9});
        //JavaRDD<String> trainingData = tmp[0]; // training set
        //System.out.println(trainingData.count());
        JavaRDD<LabeledPoint> trainingData = datatrain.map(line -> {
            String[] params = line.split(COMMA_DELIMITER);
            double label = Double.valueOf(params[2]);
            double[] vector = new double[4];
            vector[0] = Double.valueOf(params[3]);
            vector[1] = "male".equals(params[5]) ? 1d : 0d;
            vector[2] = Double.valueOf(params[6]);
            vector[3] = Double.valueOf(params[10]);
            return new LabeledPoint(label, new DenseVector(vector));
        });

        JavaRDD<LabeledPoint> testData = datatest.map(line -> {
            String[] params = line.split(COMMA_DELIMITER);
            double label = Double.valueOf(params[2]);
            double[] vector = new double[4];
            vector[0] = Double.valueOf(params[3]);
            vector[1] = "male".equals(params[5]) ? 1d : 0d;
            vector[2] = Double.valueOf(params[6]);
            vector[3] = Double.valueOf(params[10]);
            return new LabeledPoint(label, new DenseVector(vector));
        });

        // Train a GradientBoostedTrees model.
        // The defaultParams for Classification use LogLoss by default.
        BoostingStrategy boostingStrategy = BoostingStrategy.defaultParams("Classification");
        boostingStrategy.setNumIterations(3); // Note: Use more iterations in practice.
        boostingStrategy.getTreeStrategy().setNumClasses(2);
        boostingStrategy.getTreeStrategy().setMaxDepth(5);
        // Empty categoricalFeaturesInfo indicates all features are continuous.
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
        boostingStrategy.treeStrategy().setCategoricalFeaturesInfo(categoricalFeaturesInfo);

        final GradientBoostedTreesModel model = GradientBoostedTrees.train(trainingData, boostingStrategy);

        // Evaluate model on test instances and compute test error
        JavaPairRDD<Object, Object> predictionAndLabel =
                testData.mapToPair( p ->
                        (new Tuple2<>(model.predict(p.features()), p.label()))
                );
        BinaryClassificationMetrics metrics =
                new BinaryClassificationMetrics(predictionAndLabel.rdd());
        Double testErr =
                1.0 * predictionAndLabel.filter(pl -> (!pl._1().equals(pl._2()))
                ).count() / testData.count();
        Double accuracy = 1 - testErr;
        System.out.println("Accuracy: " + accuracy);

        //Confusion Matrix and Statistics
        //long[][] matrix = new long[2][2];
        long truePositive = predictionAndLabel.filter(pl -> (pl._1()).equals(pl._2()) && pl._1().equals(1d)).count();
        System.out.println("TruePositive: " + truePositive);
        //matrix[1][1] = truePositive;
        long trueNegative = predictionAndLabel.filter(pl -> (pl._1()).equals(pl._2()) && !pl._1().equals(1d)).count();
        System.out.println("TrueNegative: " + trueNegative);
        //matrix[0][0] = trueNegative;
        long falsePositive = predictionAndLabel.filter(pl -> !(pl._1()).equals(pl._2()) && pl._1().equals(1d)).count();
        System.out.println("FalsePositive: " + falsePositive);
        //matrix[1][0] = falsePositive;
        long falseNegative = predictionAndLabel.filter(pl -> !(pl._1()).equals(pl._2()) && !pl._1().equals(1d)).count();
        System.out.println("FalseNegative: " + falseNegative);
        /*matrix[0][1] = falseNegative;
        System.out.println("Confustion Matrix: " );
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                System.out.print(matrix[i][j] + " ");
            }
            System.out.println();
        }*/

        //AUC number
        System.out.println("Area under ROC = " + metrics.areaUnderROC());
    }
}