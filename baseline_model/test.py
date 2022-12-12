from transformers import AutoModel, AutoTokenizer

model_name = "bert-base-cased"

model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

if __name__ == '__main__':
    inputs = tokenizer.encode_plus("Hello world!, Today is a good day.", add_special_tokens=True,
                                   max_length=160,
                                   return_token_type_ids=False,
                                   pad_to_max_length=True,
                                   return_attention_mask=True,
                                   return_tensors='pt', )
    print(inputs.keys())
    inputs1 = tokenizer.encode("Hello world!",
                               max_length=160,
                               return_token_type_ids=False,
                               pad_to_max_length=True,
                               return_attention_mask=True,
                               return_tensors='pt',
                               output_hidden_states=True)
    inputs2 = tokenizer("Hello world!", return_tensors="pt")
    # print(inputs)
    # print(inputs1)
    # print(inputs2)
    _, outputs = model(input_ids=inputs['input_ids'],
                       attention_mask=inputs['attention_mask'], return_dict=False)
    print(inputs)
    print(_.shape)
    print(type(_))
    # print(outputs.last_hidden_state[0, 0, :])
    # print(outputs[0][:, 0, :])
