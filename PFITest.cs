using System;
using MlKombin;
using Microsoft.ML;

namespace MlKombin
{
	public class PFITest
	{
		public static void Run()
		{
			var mlContext = new MLContext();

			// Veriyi yükle
			var data = MLModel1.LoadIDataViewFromFile(
				mlContext,
				MLModel1.RetrainFilePath,
				MLModel1.RetrainSeparatorChar,
				MLModel1.RetrainHasHeader,
				MLModel1.RetrainAllowQuoting
			);

			// Modeli eğit
			var model = MLModel1.RetrainModel(mlContext, data);

			// PFI hesapla
			var pfiResults = MLModel1.CalculatePFI(mlContext, data, model, "isGoodCombo");

			// Sonuçları yazdır
			Console.WriteLine("=== Özellik Önem Değerleri (PFI) ===");
			foreach (var feature in pfiResults)
			{
				Console.WriteLine($"Özellik: {feature.Item1}, Önem: {feature.Item2:F4}");
			}
		}
	}
}