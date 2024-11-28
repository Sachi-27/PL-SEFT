import json
import pandas as pd

GRADER_NAME = "sachi"
results_file_path = "mgsm_en_llama3.2-3b-llama3.2_instruct_orig.json"
graded_file_path = results_file_path.split(".json")[0]+"_graded-by-"+GRADER_NAME+".json"
with open(results_file_path, 'r', encoding='utf-8') as f:
    results = json.load(f)

print("Total of {} results".format(len(results)))

for i in range(len(results)):
    result = results[i]
    print("QUESTION")
    print(result["question"])
    print("****************************************")
    print("ANSWER")
    print(result["output"])
    print("****************************************")
    print("CORRECT ANSWER")
    print(result["answer"])
    while True:
        response = input("Is it correct?[yes/no]").strip()
        if response == "yes" or response == "no":
            result["correct"] = 1 if response == "yes" else 0
            break
    print(""*50)
    print("#"*50)

with open(graded_file_path, 'w') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

# delete oldfile
import os
os.remove(results_file_path)
        
