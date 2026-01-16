import subprocess
import argparse

SEQ_LIST = [1024]
BATCH_LIST = [8]

def run_once(python_script, fixed_args, batch, seq,profile_out):
    """Run training script once with batch + seq."""
    cmd = ["/dev/shm/nsight-systems-2025.6.1/bin/nsys","profile","--force-overwrite","true","--cuda-um-gpu-page-faults","true" ,"--cuda-um-cpu-page-faults","true","--trace","cuda,nvtx,osrt","--cuda-memory-usage","true","--output",profile_out,
        "python", python_script,
        "--batch_size", str(batch),
        "--seq_len", str(seq),
    ] + fixed_args

    print("\n====================================================")
    print(f"Running: batch_size={batch}, seq_len={seq}")
    print("CMD:", " ".join(cmd))
    print("====================================================\n")

    subprocess.run(cmd, check=False)
    json_cmd=["/dev/shm/nsight-systems-2025.6.1/bin/nsys", 'stats', '--format','json','--report','um_sum','--output',f'{profile_out}', f'{profile_out}.nsys-rep']
    subprocess.run(json_cmd, check=False)
    
    cmd=["/dev/shm/nsight-systems-2025.6.1/bin/nsys","export","--type","sqlite","-o",f"{profile_out}",f'{profile_out}.nsys-rep']
    subprocess.run(cmd, check=False)


    

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--python_script", type=str, required=True,
                        help="Your training script, e.g., train.py")

   
    parser.add_argument("fixed", nargs=argparse.REMAINDER,
                        help="All other args passed as-is to the Python script")

    args = parser.parse_args()

   
    fixed_args = []
    skip_keys = {"--batch_size", "--seq_len"}

    i = 0
    profile_out="nsysReport__"
    while i < len(args.fixed):
        
        if args.fixed[i].split("=")[0]=='model_name' or args.fixed[i].split("=")[0]=='act_mem_pinned':
            profile_out+=(args.fixed[i].split("=")[1].replace("/", "-")+"__")
        else:
            profile_out+=(args.fixed[i].split("=")[0].replace("/", "-")+"__")
        
        if args.fixed[i] in skip_keys:
            
            i += 2
            continue
        
        fixed_args.append("--"+args.fixed[i])
        i += 1

   
    for batch in BATCH_LIST:
        for seq in SEQ_LIST:
            profile_out+=f"{batch}__{seq}"
            run_once(args.python_script, fixed_args, batch,seq,profile_out)

if __name__ == "__main__":
    main()
