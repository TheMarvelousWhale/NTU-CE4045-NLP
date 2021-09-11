import os
from flask import Flask, jsonify, request
from transformers import AutoTokenizer, AutoModelForPreTraining ,AutoConfig
import torch
import time

app = Flask(__name__)

model_path = "./../model"

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


@app.route('/', methods=['POST'])
def classify_review():
    textjson = request.json
    review = textjson['review']
    api_key = textjson['api_key']
    num_reviews = textjson['num_reviews']
    print(review, api_key)
    if review is None or api_key != "MyCustomerApiKey":
        return jsonify(code=403, message="bad request")
    start_time = time.time()
    prompt = SPECIAL_TOKENS['bos_token'] + review + SPECIAL_TOKENS['sep_token']
    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    model.eval()
    start_time = time.time()
      # Top-p (nucleus) text generation (10 samples):
    sample_outputs = model.generate(generated, 
                                      do_sample=True,   
                                      min_length=32, 
                                      max_length=128,
                                      top_k=30,                                 
                                      top_p=0.7,        
                                      temperature=0.9,
                                      repetition_penalty=2.0,
                                      num_return_sequences=num_reviews
                                      )

    texts = []
    for i, sample_output in enumerate(sample_outputs):
            text = tokenizer.decode(sample_output, skip_special_tokens=True)   
            texts.append(text[len(review):])
    elapsed_time = time.time()-start_time
    output_dict = {'prompt':review, 'result':texts,'timestamp': time.time(), 'elapsed_time': elapsed_time}
    return jsonify(output_dict)


if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google Cloud
    # Run, a webserver process such as Gunicorn will serve the app.
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))