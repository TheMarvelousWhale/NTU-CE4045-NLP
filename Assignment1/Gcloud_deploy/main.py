import os
from flask import Flask, jsonify, request
from transformers import AutoTokenizer, AutoModelForPreTraining ,AutoConfig
import torch

app = Flask(__name__)

model_path = "./model"

model_output = model_path + '/pretty-lion-15.pt'


SPECIAL_TOKENS  = { "bos_token": "<|BOS|>",
                    "eos_token": "<|EOS|>",
                    "unk_token": "<|UNK|>",                    
                    "pad_token": "<|PAD|>",
                    "sep_token": "<|SEP|>"}

tokenizer = AutoTokenizer.from_pretrained(model_path) 
tokenizer.add_special_tokens(SPECIAL_TOKENS)

model_config = AutoConfig.from_pretrained(model_path+'/config.json', 
                                    bos_token_id=tokenizer.bos_token_id,
                                    eos_token_id=tokenizer.eos_token_id,
                                    sep_token_id=tokenizer.sep_token_id,
                                    pad_token_id=tokenizer.pad_token_id,
                                    output_hidden_states=False)

model = AutoModelForPreTraining.from_pretrained(model_output, config=model_config)


@app.route('/')
def classify_review():
    review = request.args.get('review')
    api_key = request.args.get('api_key')
    if review is None or api_key != "MyCustomerApiKey":
        return jsonify(code=403, message="Meow Meow bad request :(")
    prompt = SPECIAL_TOKENS['bos_token'] + review + SPECIAL_TOKENS['sep_token']
    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    model.eval()

      # Top-p (nucleus) text generation (10 samples):
    sample_outputs = model.generate(generated, 
                                      do_sample=True,   
                                      min_length=50, 
                                      max_length=100,
                                      top_k=10,                                 
                                      top_p=0.7,        
                                      temperature=0.9,
                                      repetition_penalty=2.0,
                                      num_return_sequences=3
                                      )

    texts = []

    for i, sample_output in enumerate(sample_outputs):
            text = tokenizer.decode(sample_output, skip_special_tokens=True)   
            texts.append(text[len(review):])

    return jsonify({'result':texts})


if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google Cloud
    # Run, a webserver process such as Gunicorn will serve the app.
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))