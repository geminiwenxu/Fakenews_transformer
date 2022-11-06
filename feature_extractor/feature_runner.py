import json

import pandas as pd
import yaml
from pkg_resources import resource_filename
from transformers import BertTokenizer

from feature.feature_generator import ADJPD, AdjustedModulus, ADVPD, Alpha, APD, ATL, AutoBERTT, ASL
from feature.feature_generator import CurveLength, DPD, Entropy, Gini, HL, HPoint, IPD, NPD, Lambda, lmbd, NDW
from feature.feature_generator import PPD, PREPPD, Q, R1, RR, RRR, STC, Syn, TC, TypeTokenRatio, uniquegrams, VD, VPD
from feature.feature_generator import Scorer

print("Start")
bert_model = BertTokenizer.from_pretrained('bert-base-uncased')


def get_config(path: str) -> dict:
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


def main():
    config = get_config('/../config/config.yaml')
    feature_neg_file_path = resource_filename(__name__, config['feature_neg_test_file_path']['path'])

    df = pd.read_csv(feature_neg_file_path, sep=';')
    result = df.drop("Unnamed: 0", axis=1)
    data = []

    for index, row in result.iterrows():
        dict = {}
        text_input = row['test_input']
        tokens = bert_model.tokenize(text_input)
        # print(len(tokens))
        if len(tokens) < 510:
            try:
                sc = Scorer(scorers=[ADJPD(), AdjustedModulus(), ADVPD(), Alpha(), APD(), ATL(), AutoBERTT(), ASL(),
                                     CurveLength(), DPD(), Entropy(), Gini(), HL(), HPoint(), IPD(), NPD(), Lambda(),
                                     lmbd(), NDW(), PPD(), PREPPD(), Q(), R1(), RR(), RRR(), STC(), Syn(), TC(),
                                     TypeTokenRatio(), uniquegrams(), VD(), VPD()])
                scores, names, text_hash = sc.run("de", "dummy", text_input)
                # print(scores)
                # print(names)
                # print(text_hash)
                dict['text'] = text_input
                dict['feature'] = scores
                data.append(dict)

            except ValueError:
                print('value error')
    # print(data)
    output_file = open(resource_filename(__name__, config['neg_feature_file_path']['path']), 'w', encoding='utf-8')
    for dic in data:
        json.dump(dic, output_file)
        output_file.write("\n")


if __name__ == "__main__":
    main()
