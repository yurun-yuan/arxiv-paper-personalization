from datetime import time
from llm_summary import *

MAX_TRIES = 10
N_SEC_WAIT = 5

for i in range(MAX_TRIES):
    _, _, num_in_progress = batch_manage_llm_fetch("config.json")
    if num_in_progress == 0:
        break
    time.sleep(N_SEC_WAIT)
