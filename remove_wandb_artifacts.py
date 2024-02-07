import re

import wandb

PROJECT = "aacl_semeval"
ENTITY = "bowdbeg"
PATTERN = r".*\.pt"
api = wandb.Api(overrides={"project": PROJECT, "entity": ENTITY})

# run = api.run("bowdbeg/SemEval/d809buzh")
for run in api.runs():
    # remove files matching pattern
    for file in run.files():
        if re.match(PATTERN, file.name):
            file.delete()
            print("deleted", file.name)
