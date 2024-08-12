import json
import wandb 

# to transfer text file of attack results to wandb 
attack = "fgm"
wandb.init(project="project_name")
wandb.run.name = f"run_name_{attack}"
for i in range(1,7):
    eps = str(i/255)[0:-2]
    f = open(f'/path/to/data/wandb_run_name_{attack}_{attack}_{str(i)}.txt')
    data = json.load(f)
    wandb.log({f"loss_{attack}":data['loss'], "eps":float(eps)})
    wandb.log({f"acc1_{attack}":data['acc1'], "eps":float(eps)})
    wandb.log({f"acc5_{attack}":data['acc5'], "eps":float(eps)})