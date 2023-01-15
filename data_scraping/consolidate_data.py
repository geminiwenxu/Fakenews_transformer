import numpy as np
import pandas as pd

df_nc = pd.read_json('fake_nc_with_true.json')

# correctiv
pd_correctiv = pd.read_json('correctiv_with_true.json')

df_scrapped = pd.concat([df_nc, pd_correctiv])
# combine with fang
df_fang = pd.read_json('fang_processed.json')

# combine with
# real news
df_bild = pd.read_json('true_news_bild.json')
df_vp = pd.read_json('true_news_vp.json')
df_df = pd.read_json('true_news_df.json')
df_kz = pd.read_json('true_news_kz.json')
df_mdr = pd.read_json('true_news_mdr.json')
df_ndr = pd.read_json('true_news_ndr.json')
df_sr = pd.read_json('true_news_sr.json')

df_real = pd.concat([df_bild, df_vp, df_df, df_kz, df_mdr, df_ndr, df_sr])

# combine all
df_all = pd.concat([df_scrapped, df_fang, df_real])

# dublicates possible in fand, find them
df_all = df_all.drop_duplicates(subset='text', keep="first")
df_all = df_all.drop(['date'], axis=1)
df_all['label_id'].loc[df_all['label_id'] == 1] = '1'
df_all['label_id'].loc[df_all['label_id'] == 0] = '0'

# drop row if body is None
df_all = df_all[df_all['text'].notna()]

# drop dublicate urls and index
df_all = df_all[~df_all.index.duplicated(keep='first')]
df_all.drop_duplicates(subset='url', keep="first", inplace=True)

# make sure same label is used
df_all['label'] = np.where(df_all.label == 'true', 'real', df_all.label)

df_all.to_json(path_or_buf='../data/data_all.json', force_ascii=False, orient='records')
