MODEL_PATH=$1
MODEL_DEFINITION=$2
TARGET_MODEL_DEFINITION=./lm_eval/models/new_model.py
num_fs=${3:-"0"}
OUTPUT_PATH=${4:-"results/"$MODEL_DEFINITION}
TASKS=${5:-"arc_challenge,arc_easy,openbookqa,boolq,hellaswag,piqa,winogrande,mmlu,social_iqa,sciq"}

# Copy model file content to target path
cat /T6_MODEL_DEFINITION_DIR/$MODEL_DEFINITION.py > $TARGET_MODEL_DEFINITION

lm_eval --model hf --model_args pretrained=$MODEL_PATH --trust_remote_code --tasks $TASKS --batch_size auto --output_path $OUTPUT_PATH --num_fewshot $num_fs