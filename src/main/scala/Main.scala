import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.{Pipeline, PipelineModel}

object Main {
  def main(args: Array[String]): Unit = {

    // Initialize SparkSession
    val spark = SparkSession.builder()
      .appName("FraudAnalysis")
      .config("spark.master", "local")
      .getOrCreate()


    // Read Data From HDFS
    val hdfsFilePath = "data/FraudData.csv"
    val fraudData: DataFrame = spark.read
      .option("header", "true") // Set to "false" if your dataset doesn't have a header
      .option("inferSchema", "true") // Infer data types automatically
      .csv(hdfsFilePath)

    // Do Analysis By RiskLevel
    val RiskLevelAnalysis: DataFrame = fraudData
      .withColumn("RiskLevel", when(col("RiskLevel").isNull, "No Risk").otherwise(col("RiskLevel")))
      .groupBy("RiskLevel")
      .agg(
        count("LoanID").alias("TotalLoans"),
        sum("Fraud").alias("TotalFrauds"),
        avg("AnnualIncome").alias("AvgIncome"),
        avg("CreditScore").alias("AvgCreditScore"),
        max("CustomerAge").alias("MaxCustomerAge"),
        min("CustomerAge").alias("MinCustomerAge")
      )
    RiskLevelAnalysis.show()

    /* Save Result into HDFS
    RiskLevelAnalysis.coalesce(1)
      .write.mode("overwrite")
      .option("header", "true")
      .csv("Output/RiskLevelAnalysis")*/

    // Calculate Average LatePaymentAmount For Fraud and Non-Frauds Using sql
    fraudData.createOrReplaceTempView("FraudData")
    val LatePaymentFraudAnalysis: DataFrame = spark.sql(
      "Select Fraud, AVG(LatePaymentAmount) as AverageLatePaymentAmount From FraudData Group By Fraud")
    LatePaymentFraudAnalysis.show()
    /* Save Result into HDFS
    LatePaymentFraudAnalysis.coalesce(1)
      .write.mode("overwrite")
      .option("header", "true")
      .csv("Output/LatePaymentFraudAnalysis") */

    // Machine Learning Model
    // Prepare the data for classification
    val assembler = new VectorAssembler()
      .setInputCols(Array("CustomerAge", "LatePaymentAmount"))
      .setOutputCol("features")

    // Create the Logistic Regression classifier
    val lr = new LogisticRegression()
      .setLabelCol("Fraud")
      .setFeaturesCol("features")
      .setMaxIter(10)

    // Create a Pipeline to chain the VectorAssembler and LogisticRegression
    val pipeline = new Pipeline()
      .setStages(Array(assembler, lr))

    // Split the data into training and testing sets (80% for training, 20% for testing)
    val Array(trainingData, testData) = fraudData.randomSplit(Array(0.8, 0.2), seed = 42)

    // Train the model using the Pipeline
    val model: PipelineModel = pipeline.fit(trainingData)

    // Make predictions on the test set
    val predictions = model.transform(testData)

    // Evaluate the model using BinaryClassificationEvaluator for both AUC and accuracy
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("Fraud")
      .setRawPredictionCol("rawPrediction")

    val au_roc = evaluator.setMetricName("areaUnderROC").evaluate(predictions)
    val au_pr = evaluator.setMetricName("areaUnderPR").evaluate(predictions)


    // Save the evaluation results to HDFS
    val evaluationResult = spark
      .createDataFrame(Seq(("Area Under ROC", au_roc), ("Area Under PR", au_pr)))
      .toDF("Metric", "Value")
    evaluationResult.show()
    /*evaluationResult.coalesce(1)
      .write.mode("overwrite")
      .option("header", "true")
      .csv("Output/ModelEvaluation")*/

    spark.stop()
  }
}