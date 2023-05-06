from header import *

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--model_path', default='../ckpt/openllama', type=str)
    parser.add_argument('--max_length', default=512, type=int)
    parser.add_argument('--generate_len', default=512, type=int)
    parser.add_argument('--top_k', default=50, type=int)
    parser.add_argument('--top_p', default=0.92, type=float)
    return parser.parse_args()


def main(args):
    model = LlamaForCausalLM.from_pretrained(args['model_path']).cuda()
    tokenizer = LlamaTokenizer.from_pretrained(args['model_path'])

    # instruction = 'What is the Natural Language Processing'
    instruction = input('[!] Input your instruction: ')
    prompt_no_input = f'### Instruction:\n{instruction}\n\n### Response:'
    tokens = tokenizer.encode(prompt_no_input)
    tokens = [1] + tokens + [2] + [1]
    tokens = torch.LongTensor(tokens[-args['max_length']+args['generate_len']:]).unsqueeze(0).cuda()

    length = len(tokens[0])
    with torch.no_grad():
        rest = model.generate(
            input_ids=tokens, 
            max_length=length+args['generate_len'], 
            use_cache=True, 
            do_sample=True, 
            top_p=args['top_p'], 
            top_k=args['top_k']
        )
    output = rest[0][length:]
    string = tokenizer.decode(output, skip_special_tokens=False)
    string = string.replace('<s>', '').replace('</s>', '').strip()
    print(f'[!] OpenAlpaca generation: {string}')

if __name__ == "__main__":
    args = vars(parser_args())
    main(args)
