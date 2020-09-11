class DialoGPT2(object):
    def __init__(self, model_name_or_path, cuda_device=None, use_context=False):
        import torch
        from transformers import GPT2Tokenizer, GPT2LMHeadModel
        self.device = torch.device('cuda:' + args.cuda_device if cuda_device is not None and torch.cuda.is_available() else 'cpu')
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
        self.use_context = use_context
        self.reset_context()

    def reset_context(self):
        self.chat_history_ids = None

    def gen(self, inp):
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = self.tokenizer.encode(inp + self.tokenizer.eos_token, return_tensors='pt').to(self.device)


        # append the new user input tokens to the chat history
        bot_input_ids = (torch.cat([self.chat_history_ids, new_user_input_ids], dim=-1).to(self.device) if (self.use_context and self.chat_history_ids is not None) else new_user_input_ids)

        # generated a response while limiting the total chat history to 1000 tokens, 
        self.chat_history_ids = self.model.generate(
            bot_input_ids, max_length=200,
            pad_token_id=self.tokenizer.eos_token_id,  
            no_repeat_ngram_size=3,       
            do_sample=True, 
            top_k=100, 
            top_p=0.7,
            temperature = 0.8
        )
        
        # pretty print last ouput tokens from bot
        return "{}".format(self.tokenizer.decode(self.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True))


