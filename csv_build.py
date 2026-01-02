import os
import pandas as pd


CSV_FOLDER = "."     # folder containing all experiment CSVs
MASTER_CSV = "master.csv"       # final combined CSV


'''FILENAME_COLUMNS = [
    "model_name",
    "seq_len",
    "steps",
    "batch_size",
    "prefetch_layers",
    "optimisation",
    "prefetching",
    "num_layer_pinned",
    "act_mem_pinned",
    "weight_pinned",
    "optimiser_prefetch",
    "optimiser_offload",
    "activation_prefetch",
    "logging",
    "backward_prefetch",
    "build_csv",
]'''

FILENAME_COLUMNS = [
    "model_name",
    "seq_len",
    "steps",
    "batch_size",
    "prefetch_layers",
    "optimisation",
    "prefetching",
    "num_layer_pinned",
    "act_mem_pinned",
    "weight_pinned",
    "optimiser_prefetch",
    "optimiser_offload",
    "activation_prefetch",
    "logging",
    "backward_prefetch",
    "weight_prefetch",
    "build_csv",
]

all_dfs = []

for filename in os.listdir(CSV_FOLDER):
    if not filename.endswith(".csv"):
        continue

    filepath = os.path.join(CSV_FOLDER, filename)

    
    name = filename[:-4]                 # strip ".csv"
    parts = name.split("-")

    if len(parts) < len(FILENAME_COLUMNS) and parts[-13]=="8":
        print(f"⚠️ Skipping malformed filename: {filename}")
        continue

    
    if parts[-13]=="8":

        model_name = "-".join(parts[:-15])

        fixed_tail = parts[-15:]

        '''if model_name != "ibm-granite-granite-3.0-8b-base":
            #print(model_name)
            #print("check")
            continue'''
        fixed_tail.append("1.0")
        fixed_tail[-2]="0"
        metadata = dict(zip(
            FILENAME_COLUMNS,
            [model_name] + fixed_tail
        ))
    else:
        model_name = "-".join(parts[:-16])
        fixed_tail = parts[-16:]
        '''if model_name != "ibm-granite-granite-3.0-8b-base":
            #print(model_name)
            #print("check")
            continue'''

        metadata = dict(zip(
            FILENAME_COLUMNS,
            [model_name] + fixed_tail
        ))


    
    df = pd.read_csv(filepath)

    
    for col, val in metadata.items():
        df[col] = [val]*df.shape[0]

    
    df["source_csv"] = [filename]*df.shape[0]

    all_dfs.append(df)


if not all_dfs:
    raise RuntimeError("❌ No valid CSV files found")

master_df = pd.concat(all_dfs, ignore_index=True, sort=False)


master_df.to_csv(MASTER_CSV, index=False)

print(f"✅ Master CSV successfully created: {MASTER_CSV}")