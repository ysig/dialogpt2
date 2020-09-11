## Statement

I transformed this [notebook](https://towardsdatascience.com/make-your-own-rick-sanchez-bot-with-transformers-and-dialogpt-fine-tuning-f85e6d1f4e30) into a package.

## Requirements

Install [huggingface](https://github.com/huggingface/transformers) :)

## Install this repo

`pip install dialogpt2`

## Usage

There are two scripts and a class.

### Train

```bash
$ dialogpt2-train --help
usage: dialogpt2-train [-h] --input_file INPUT_FILE [--line_sep LINE_SEP]
                       [--qa_sep QA_SEP] [--output_dir OUTPUT_DIR]
                       [--model_name_or_path MODEL_NAME_OR_PATH]
                       [--config_name CONFIG_NAME]
                       [--tokenizer_name TOKENIZER_NAME]
                       [--cache_dir CACHE_DIR] [--block_size BLOCK_SIZE]
                       [--do_train] [--do_eval] [--evaluate_during_training]
                       [--per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE]
                       [--per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE]
                       [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
                       [--learning_rate LEARNING_RATE]
                       [--weight_decay WEIGHT_DECAY]
                       [--adam_epsilon ADAM_EPSILON]
                       [--max_grad_norm MAX_GRAD_NORM]
                       [--num_train_epochs NUM_TRAIN_EPOCHS]
                       [--max_steps MAX_STEPS] [--warmup_steps WARMUP_STEPS]
                       [--logging_steps LOGGING_STEPS]
                       [--save_steps SAVE_STEPS]
                       [--save_total_limit SAVE_TOTAL_LIMIT]
                       [--eval_all_checkpoints] [--no_cuda]
                       [--overwrite_output_dir] [--overwrite_cache]
                       [--should_continue] [--seed SEED]
                       [--local_rank LOCAL_RANK] [--fp16]
                       [--fp16_opt_level FP16_OPT_LEVEL]

optional arguments:
  -h, --help            show this help message and exit
  --input_file INPUT_FILE, -i INPUT_FILE
                        Input file is a list lines that contain a single
                        question and a single answer.
  --line_sep LINE_SEP   Line separation token
  --qa_sep QA_SEP       Token that separates question with an answer
  --output_dir OUTPUT_DIR
                        Output-dir of the model
  --model_name_or_path MODEL_NAME_OR_PATH
  --config_name CONFIG_NAME
  --tokenizer_name TOKENIZER_NAME
  --cache_dir CACHE_DIR
  --block_size BLOCK_SIZE
  --do_train
  --do_eval
  --evaluate_during_training
  --per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE
  --per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
  --learning_rate LEARNING_RATE
  --weight_decay WEIGHT_DECAY
  --adam_epsilon ADAM_EPSILON
  --max_grad_norm MAX_GRAD_NORM
  --num_train_epochs NUM_TRAIN_EPOCHS
  --max_steps MAX_STEPS
  --warmup_steps WARMUP_STEPS
  --logging_steps LOGGING_STEPS
  --save_steps SAVE_STEPS
  --save_total_limit SAVE_TOTAL_LIMIT
  --eval_all_checkpoints
  --no_cuda
  --overwrite_output_dir
  --overwrite_cache
  --should_continue
  --seed SEED
  --local_rank LOCAL_RANK
  --fp16
  --fp16_opt_level FP16_OPT_LEVEL

```


### Gen

```bash
$ dialogpt2-gen -i --help
usage: dialogpt2-gen [-h]
                     (--question QUESTION | --questions-file QUESTIONS_FILE | --interactive)
                     [--num_outputs NUM_OUTPUTS] [--cuda-device CUDA_DEVICE]
                     [--model_name_or_path MODEL_NAME_OR_PATH] [--use_context]
                     [--reset_context_on_iter]

optional arguments:
  -h, --help            show this help message and exit
  --question QUESTION, -q QUESTION
                        A single question.
  --questions-file QUESTIONS_FILE, -qf QUESTIONS_FILE
                        Questions separated by a line break character.
  --interactive, -i     Talk with it!
  --num_outputs NUM_OUTPUTS, -n NUM_OUTPUTS
  --cuda-device CUDA_DEVICE, -c CUDA_DEVICE
  --model_name_or_path MODEL_NAME_OR_PATH, -m MODEL_NAME_OR_PATH
  --use_context, -uc
  --reset_context_on_iter, -rcoi
                        Whne num_outputs is bigger than 1 reset context on
                        each output batch

```

### DialoGPT


```python
from dialogpt2 import DialoGPT2
gpt = DialoGPT2(model_name_or_path='microsoft/DialoGPT-small', cuda_device=None, use_context=False)
gpt.gen('Could you image someone building a repo about you?')
```

### Example

```bash
$ dialogpt2-gen -i
User >> Who made you?
Bot >> I'm not sure I follow...
User >> Shame
Bot >> You are a very sad person.
User >> Don't judge so fast 
Bot >> The judge said that?
User >> Shut up 
Bot >> What a nice way to get a ring.
User >> AAAAA
Bot >> AA
User >> A
Bot >> 
User >> AAAAAAAAAAAAAAA
Bot >> He's a good looking guy.
User >> 
```
