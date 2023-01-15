import glob

import pandas as pd
import yaml
from pkg_resources import resource_filename

from data_scraping.get_data_statistics import get_source


def get_config(path: str) -> dict:
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


def main():
    config = get_config('/../config/config.yaml')
    df_data = pd.read_json(resource_filename(__name__, config['test']['path']))

    wrongs = glob.glob('wrong*')

    # get source proportions for all data
    df_source_all = get_source(df_data)
    df_source_all = df_source_all['source'].value_counts(normalize=True).to_frame().reset_index(level=0)
    df_source_all = df_source_all.rename(columns={'index': 'source', 'source': 'proportion_data'})

    sources_count = []
    for wrong in wrongs:
        df_wrong = pd.read_json(wrong)
        # join
        df_merged = pd.merge(df_wrong, df_data, how='inner', on='text')
        df_source = get_source(df_merged)
        df_source = df_source['source'].value_counts(normalize=True).to_frame().reset_index(level=0)
        df_source['model'] = wrong.replace('.json', '').replace('wrong_', '')
        df_source = df_source.rename(columns={'index': 'source', 'source': 'proportion_wrong'})
        sources_count.append(df_source)

    appended_data = pd.concat(sources_count)
    df_merged_counts = pd.merge(appended_data, df_source_all, how='inner', on='source')
    df_merged_counts['proportion_data'] = (df_merged_counts['proportion_data'] * 100).round(2)
    df_merged_counts['proportion_wrong'] = (df_merged_counts['proportion_wrong'] * 100).round(2)

    # get delta
    df_merged_counts['Abweichung'] = (df_merged_counts['proportion_wrong'] - df_merged_counts['proportion_data']).round(
        2)

    # change column order for latex export

    df_merged_counts[['source', 'model', 'proportion_wrong', 'proportion_data', 'Abweichung']].to_latex(
        "analysis_wrong.txt", index=False)


if __name__ == "__main__":
    main()
