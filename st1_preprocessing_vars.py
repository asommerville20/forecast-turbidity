# import packages
from dataretrieval import nwis
import pandas as pd

# begin turbidity importation process
site = '11447650' # specify USGS site code
parameter_code = '63680' # turbidity (FNU)
start_date = '1948-01-01'
end_date = '2025-08-08'

## pull instantaneous values (iv)
df, metadata = nwis.get_iv(
    sites = site,
    parameterCd = parameter_code,
    start = start_date,
    end = end_date
)

## filter df to just relevant columns
df_filtered = pd.DataFrame()
for col in df.columns:
    if '63680' in col:
        df_filtered[col] = df[col]
    elif 'datetime' in col:
        df_filtered[col] = df[col]

## turbidity measured with two different instruments non-concurrently -- must merge
df_filtered['turb'] = df_filtered[
    '63680_median ts087: ysi model 6136'].combine_first(
    df_filtered['63680_bgc project, [east fender']
)
df_filtered['turb_cd'] = df_filtered[
    '63680_median ts087: ysi model 6136_cd'].combine_first(
    df_filtered['63680_bgc project, [east fender_cd']
)

## drop old columns
drop_cols = [
    '63680_median ts087: ysi model 6136',
    '63680_bgc project, [east fender',
    '63680_median ts087: ysi model 6136_cd',
    '63680_bgc project, [east fender_cd'
]
df_filtered.drop(columns = drop_cols, 
                 axis = 1, 
                 inplace = True)
