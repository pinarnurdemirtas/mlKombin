using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;

class Program
{
public class ModelInput
{
	[LoadColumn(0)]
	[ColumnName("upperCategory")]
	public string UpperCategory { get; set; }

	[LoadColumn(1)]
	[ColumnName("upperColor")]
	public string UpperColor { get; set; }

	[LoadColumn(2)]
	[ColumnName("bottomCategory")]
	public string BottomCategory { get; set; }

	[LoadColumn(3)]
	[ColumnName("bottomColor")]
	public string BottomColor { get; set; }

	[LoadColumn(4)]
	[ColumnName("shoesCategory")]
	public string ShoesCategory { get; set; }

	[LoadColumn(5)]
	[ColumnName("shoesColor")]
	public string ShoesColor { get; set; }

	[LoadColumn(6)]
	[ColumnName("season")]
	public string Season { get; set; }

	[LoadColumn(7)]
	[ColumnName("temperature")]
	public float Temperature { get; set; }

	[LoadColumn(8)]
	[ColumnName("isGoodCombo")]
	public bool IsGoodCombo { get; set; }
}


static void Main(string[] args)
	{
		var mlContext = new MLContext();
		var dataPath = @"C:\model\yeni_model.mlnet";

		var data = mlContext.Data.LoadFromTextFile<ModelInput>(
			dataPath, hasHeader: true, separatorChar: ',');

		var pipeline = mlContext.Transforms.Categorical.OneHotEncoding(new[]
		{
			new InputOutputColumnPair("upperCategory"),
			new InputOutputColumnPair("upperColor"),
			new InputOutputColumnPair("bottomCategory"),
			new InputOutputColumnPair("bottomColor"),
			new InputOutputColumnPair("shoesCategory"),
			new InputOutputColumnPair("shoesColor"),
			new InputOutputColumnPair("season")
		})
		.Append(mlContext.Transforms.ReplaceMissingValues("temperature"))
		.Append(mlContext.Transforms.Concatenate("Features", new[]
		{
			"upperCategory","upperColor","bottomCategory","bottomColor",
			"shoesCategory","shoesColor","season","temperature"
		}));

		var trainData = pipeline.Fit(data).Transform(data);

		var losses = new List<string>();

		for (int i = 1; i <= 50; i++) // 50 epoch gibi düşünebilirsin
		{
			var options = new LightGbmBinaryTrainer.Options
			{
				NumberOfIterations = i,
				LabelColumnName = "IsGoodCombo",
				FeatureColumnName = "Features",
				LearningRate = 0.1
			};

			var trainer = mlContext.BinaryClassification.Trainers.LightGbm(options);
			var model = trainer.Fit(trainData);

			var predictions = model.Transform(trainData);
			var metrics = mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: "IsGoodCombo");

			Console.WriteLine($"Epoch {i} - LogLoss: {metrics.LogLoss}");
			losses.Add($"{i},{metrics.LogLoss}");
		}

		File.WriteAllLines("logloss.csv", losses);
		Console.WriteLine("logloss.csv dosyası oluşturuldu.");
	}
}
