from header import *
import gradio as gr
import mdtex2html

#########################################################################
####################### gradio utils functions ##########################
#### code borrowed from ChatGLM: https://github.com/THUDM/ChatGLM-6B ####
####################### gradio utils functions ##########################
#########################################################################

def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text

def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y

def reset_user_input():
    return gr.update(value='')


def reset_state():
    return []


#########################################################################
####################### gradio utils functions ##########################
#### code borrowed from ChatGLM: https://github.com/THUDM/ChatGLM-6B ####
####################### gradio utils functions ##########################
#########################################################################

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--model_path', default='../ckpt/openllama', type=str)
    return parser.parse_args()


def encode(vocab, instruction, max_length=1024, generation_length=128):
    prompt_no_input = f'### Instruction:\n{instruction}\n\n### Response:'
    tokens = tokenizer.encode(prompt_no_input)
    tokens = [1] + tokens + [2] + [1]
    tokens = torch.LongTensor(tokens[-max_length+generation_length:]).unsqueeze(0).cuda()
    return tokens

def chat(input, chatbot, max_length, gen_length, top_k, top_p):
    input_ids = encode(tokenizer, input, generation_length=gen_length, max_length=max_length)
    length = len(input_ids[0])
    with torch.no_grad():
        rest = model.generate(
            input_ids=input_ids,
            max_length=length + gen_length,
            use_cache=True,
            do_sample=True,
            top_p=top_p,
            top_k=top_k
        )
        output = rest[0][length:]
        string = tokenizer.decode(output, skip_special_tokens=False)
        string = string.replace('<s>', '').replace('</s>', '').strip()
    chatbot.append((parse_text(input), parse_text(string)))
    return chatbot

if __name__ == "__main__":
    args = vars(parser_args())
    gr.Chatbot.postprocess = postprocess 
    # init model
    model = LlamaForCausalLM.from_pretrained(args['model_path']).cuda()
    tokenizer = LlamaTokenizer.from_pretrained(args['model_path'])
    print(f'[!] load model over ...')
    # gradio interface
    with gr.Blocks() as demo:
        gr.HTML("""<h1 align="center">OpenAlpaca</h1>""")
        chatbot = gr.Chatbot()
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Column(scale=12):
                    user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(container=False)
                with gr.Column(min_width=32, scale=1):
                    submitBtn = gr.Button("Submit", variant="primary")
            with gr.Column(scale=1):
                emptyBtn = gr.Button("Clear History")
                max_length = gr.Slider(0, 1024, value=512, step=1.0, label="Maximum length", interactive=True)
                generate_len = gr.Slider(0, 512, value=256, step=1.0, label="Maximum length", interactive=True)
                top_p = gr.Slider(0.01, 1, value=0.75, step=0.01, label="Top P", interactive=True)
                top_k = gr.Slider(1, 100, value=50, step=1, label="Top K", interactive=True)
        submitBtn.click(
            chat, [
                user_input, 
                chatbot, 
                max_length, 
                generate_len,
                top_k, 
                top_p
            ], [
                chatbot
            ],
        )
        submitBtn.click(reset_user_input, [], [user_input])
        emptyBtn.click(reset_state, outputs=[chatbot])

    demo.launch(server_name='0.0.0.0', server_port=23001, share=False, debug=True)  
