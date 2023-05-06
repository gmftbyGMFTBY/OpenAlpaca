from header import *

class OpenLLAMAModel(nn.Module):

    def __init__(self, **args):
        super(OpenLLAMAModel, self).__init__()
        self.args = args

        # init the tokenizer
        self.vocab = LlamaTokenizer.from_pretrained(args['model_path'])
        self.vocab.bos_token_id = 1
        self.vocab.eos_token_id = 2

        self.model = LlamaForCausalLM.from_pretrained(args['model_path'], torch_dtype=torch.float16)
        self.model.cuda(torch.cuda.current_device())
        total = sum([param.nelement() for param in self.parameters()])
        print('[!] Model Size: %2fB' % (total/1e9))

    def forward(self, inputs):
        labels = inputs['labels'].cuda()
        outputs = self.model(
            input_ids=inputs['input_ids'].cuda(), 
            attention_mask=inputs['attention_mask'].cuda(), 
            labels=labels
        )
        loss = outputs.loss
        
        # calculate the token accuarcy
        logits = outputs.logits[:, :-1, :]
        labels = labels[:, 1:]
        chosen_tokens = torch.max(logits, dim=-1)[1]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask
        gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()
        return loss, gen_acc
 
