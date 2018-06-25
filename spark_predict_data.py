from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.sql.functions import col

from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.classification import LogisticRegression

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

from pyspark.ml.evaluation import BinaryClassificationEvaluator,MulticlassClassificationEvaluator
from pyspark.ml import PipelineModel


sc =SparkContext()
sqlContext = SQLContext(sc)
model = PipelineModel.load("logreg.model")
print (type(model))

tweet_data = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('all_results.csv')
predictions = model.transform(tweet_data)
# predictions.show(5)
# print (type(predictions))
predictions.select("SentimentText", "filtered", "prediction").show(6)
# predictions.write.csv('mycsv.csv')
predictions.select("SentimentText", "prediction").toPandas().to_csv('predict_sentiment.csv')

