import json
import sys
def read_and_deduplicate(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            address_size_list = {}
            for line in lines:
                line = line.strip()
                if line:
                    parts = line.split(', ')
                    address = parts[0].split(': ')[1]
                    size = int(parts[1].split(': ')[1])
                    #address_size_list.append((address, size))
                    address_size_list[address]=size
            #deduplicated_list = list(set(address_size_list))

            return address_size_list

    except FileNotFoundError:
        print("File not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def read_optimizer_state(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            optimizer_state_list = []
            for line in lines:
                line = line.strip()
                if line:
                    parts = line.split(', ')
                    address = parts[1].split(': ')[1]
                    size = int(parts[2].split(': ')[1])
                    optimizer_state_list.append((address, size))

            optimizer_state_list = list(set(optimizer_state_list))
            return optimizer_state_list

    except FileNotFoundError:
        print("File not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def read_module_info(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            activation_list = []
            for line in lines:
                line = line.strip()
                if line:
                    parts = line.split(', ')
                    module_name = parts[0].split(': ')[1]
                    address = parts[1].split(': ')[1]
                    size = int(parts[2].split(': ')[1])
                    activation_list.append((address, size))
            activation_list = list(set(activation_list))
            return activation_list

    except FileNotFoundError:
        print("File not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def int_conversion(x):
    try:
          return int(x)
    except:
          return 0
def filter_json_data(file_path, key, value):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            if isinstance(data, list):
                filtered_data = [item  for item in data if key in item and int_conversion(item[key]) >= value]
            elif isinstance(data, dict):
                filtered_data = {k: v for k, v in data.items() if k == key and int_conversion(v) >=  value}
                if not filtered_data:
                    filtered_data = [data] if key in data and int_conversion(data[key]) >= value else []
            else:
                print("Invalid JSON format.")
                return None

            return filtered_data

    except FileNotFoundError:
        print("File not found.")
        return None
    except :
        print("Invalid JSON format.")
        return None



def filter_json_eviction(file_path, key, value):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)

            return data

    except FileNotFoundError:
        print("File not found.")
        return None
    except :
        print("Invalid JSON format.")
        return None

file_path = 'fault.json'
key = " GPU Page Faults"
value = 2

#Address: 0x7f2500000000, Size: 33554432
#Address: 0x7f98ee000000, Size: 33554432
#0x7f12e4000000, Size: 268435456
filtered_data = filter_json_data(file_path, key, value)
weights=read_and_deduplicate("weight.txt")
activations=read_module_info("input.txt")
gradients=read_module_info("gradient.txt")
optimizer=read_optimizer_state("optimizer.txt")
sum=0
for item in filtered_data:
   sum+=item[" GPU Page Faults"]


print(sum)
#sys.exit()
if filtered_data:
    sum_weight=0
    sum_input=0
    sum_optimizer=0
    sum_gradient=0
    fault_dict=[]
    iter=0
    for item in filtered_data:   
       for t in weights :
        
            i=int(item["Virtual Address"],16)
            if i >= int(t,16) and i<=int(t,16)+weights[t]:
                #sum_weight+=item[" GPU Page Faults"]
                if item["Virtual Address"] not in fault_dict:
                      #print("error")
                   sum_weight+=item[" GPU Page Faults"]
                   fault_dict.append(item["Virtual Address"])
                   break

       #print(sum_weight)
       for t in activations :
   
            i=int(item["Virtual Address"],16)
            if i >= int(t[0],16) and i<=int(t[0],16)+t[1] and (item["Virtual Address"] not in fault_dict):
                sum_input+=item[" GPU Page Faults"]
                fault_dict.append(item["Virtual Address"])
                break
       #print(sum_input)
 
       for t in optimizer :
    
            i=int(item["Virtual Address"],16)
            if i >= int(t[0],16) and i<=int(t[0],16)+t[1] and (item["Virtual Address"] not in fault_dict):
                sum_optimizer+=item[" GPU Page Faults"]
                fault_dict.append(item["Virtual Address"])
                break

       for t in gradients :
    
            i=int(item["Virtual Address"],16)
            if i >= int(t[0],16) and i<=int(t[0],16)+t[1] and (item["Virtual Address"] not in fault_dict):
                sum_gradient+=item[" GPU Page Faults"]
                fault_dict.append(item["Virtual Address"])
                break
       #print(sum_optimizer)
       iter=iter+1
       print(f"done : {iter} out of {len(filtered_data)}")
print(sum_weight,sum_input,sum_optimizer,sum_gradient)
