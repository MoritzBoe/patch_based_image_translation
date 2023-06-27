import wandb
import numpy as np

"""Delete saved images from wandb, since they consume a lot of space.
This can be used, when --val_frequence in lit_cyclegan.py was to high.
"""

api = wandb.Api()

run = api.run("Path_to_run")

files = run.files()
files_len = len(files)

i = 0
counter = 0
else_counter = 0
# delete backwards because files does refresh sometimes and then len is shortened.
for idx in np.arange(files_len-1, -1, -1):
    if "examples" in files[idx].name:
        if 0 < i < 20:
            files[idx].delete()
            counter = counter+1
        i = i + 1
        if i == 20:
            i = 0
    print(f"Deleted a total of {counter} files")
