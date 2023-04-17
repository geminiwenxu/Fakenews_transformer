from transformers import AutoModel, AutoTokenizer

if __name__ == '__main__':
    texts = 'When was I last outside? I am stuck at home for 2 weeks. lalalalala.'

    model = AutoModel.from_pretrained('bert-base-german-cased')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-german-cased')
    tokenized = tokenizer(texts, add_special_tokens=True,
                          max_length=160,
                          return_token_type_ids=False,
                          pad_to_max_length=True,
                          return_attention_mask=True,
                          return_tensors='pt')
    batch_size = 2

    bert_output = model.forward(**tokenized, output_hidden_states=True)
    # print(bert_output.last_hidden_state.size())

    bert = AutoModel.from_pretrained('bert-base-german-cased', output_hidden_states=True)
    encoding = bert.encode_plus(texts, add_special_tokens=True,
                                max_length=160,
                                return_token_type_ids=False,
                                pad_to_max_length=True,
                                return_attention_mask=True,
                                return_tensors='pt')
    bert_results = model(encoding['input_ids'], encoding['attention_mask'])
    # print(bert_results.last_hidden_state[:, 0, :])
    # print(bert_results[0][:, 0, :])
    print(bert_results)
    encoded_layers = bert_results['hidden_states']
    print(encoded_layers)

    # # calculate weights per batch
    # labels = torch.tensor([0, 0, 0, 1, 1]).to(torch.long)
    # n_labels = 2
    # per_batch_weights = F.one_hot(labels, n_labels).sum(0)
    # print(per_batch_weights)
    # per_batch_weights = per_batch_weights.max().div(per_batch_weights)
    # print(per_batch_weights)
