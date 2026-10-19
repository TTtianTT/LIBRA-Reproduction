# Azure Functions Invocation Trace 2021

[Microsoft dataset documentation](https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsInvocationTrace2021.md). Download: [official RAR](https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureFunctionsInvocationTraceForTwoWeeksJan2021.rar).

Attribution: Yanqi Zhang, Inigo Goiri, Gohar Irfan Chaudhry, Rodrigo Fonseca, Sameh Elnikety, Christina Delimitrou, Ricardo Bianchini. Faster and Cheaper Serverless Computing on Harvested Resources. SOSP 2021.

License: [CC BY](https://github.com/Azure/AzurePublicDataset/blob/master/LICENSE).

The dedicated schema specifies seconds. Arrival is end_timestamp minus duration; all zero-demand seconds are retained. The aggregate covers 14 days, with out-of-range arrivals counted explicitly in manifest.json. No applications were selected and no timestamps were stretched or smoothed. Original request durations are used only to reconstruct arrivals; downstream LIBRA simulation retains its fixed one-second service model.

Raw files are local under raw/ and ignored by Git. aggregate_counts.npz stores the unscaled per-second counts; manifest.json stores hashes.
