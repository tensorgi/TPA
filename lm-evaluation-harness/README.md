# lm-evaluation-harness
Cloned from EleutherAI/lm-evaluation-harness (https://github.com/EleutherAI/lm-evaluation-harness)
Edited 
First: pip install -e .

Then edit the test.sh:
 - **TARGET_MODEL_DEFINITION=./lm_eval/models/new_model.py** 
   - change it as /YOUR_DIR_TO_LM_EVALUATION_HARNESS_REPOSITORY/lm_eval/models/new_model.py for your directory to lm-evaluation-harness repo
 - **cat /T6_MODEL_DEFINITION_DIR/$MODEL_DEFINITION.py > $TARGET_MODEL_DEFINITION**
   - change the prefix to your directory of nanoGPT-neo

Then run test.sh: bash test.sh MODEL_PATH MODEL_DEFINITION num_fs OUTPUT_PATH TASKS
 - MODEL_PATH: the directory of the saved checkpoint
 - MODEL_DEFINITION: the type of model: T6, llama-mha, llama-gqa (both gqa and mqa use the same architecture), llama-mla
 - num_fs: number of few-shot, default 0
 - OUTPUT_PATH: output result json path, default: "results/"$MODEL_DEFINITION, you can change for different location
 - TASKS: tasks, default: arc_easy,arc_challenge,openbookqa,boolq,hellaswag,piqa,winogrande,mmlu,sciq
