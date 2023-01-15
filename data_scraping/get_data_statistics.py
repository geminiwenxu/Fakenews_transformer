import pandas as pd


def get_source(df_all):
    ## get different sources, use regex to get information between a doppeltpunkt and point then remove www
    # https://de.sott, https:\/\/de.rt.com are not falling into this pattern, replace these
    df_all['source_raw'] = df_all['url'].str.replace('www.', '')
    df_all['source_raw'] = df_all['source_raw'].str.replace('https://de.sott', 'https://sott')
    df_all['source_raw'] = df_all['source_raw'].str.replace('https://de.rt', 'https://rt')
    df_all['source_start'] = df_all['source_raw'].str.find(':')
    df_all['source_end'] = df_all['source_raw'].str.find('.')

    df_all['source'] = df_all.apply(lambda x: x['source_raw'][x['source_start'] + 3:x['source_end']], 1)

    return df_all


def main():
    df_all = pd.read_json('../data/data_all.json')
    # get counts
    df_source = get_source(df_all)
    print(df_source['source'].value_counts())
    df_true = df_source[df_source['label_id'] == 1]
    print(df_true.shape)
    df_fake = df_source[df_source['label_id'] == 0]
    print(df_fake.shape)


if __name__ == "__main__":
    main()
