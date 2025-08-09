# import packages
from dataretrieval import nwis
import pandas as pd

# --- 1 --- #
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

## remove rows without approval status
df_clean = df_filtered.loc[df_filtered['turb_cd'] == 'A']
df_clean.drop(columns = ['turb_cd'], inplace = True)

## extract amount of dropped rows
original_length = len(df_filtered)
approved_length = len(df_clean)
dropped_count = original_length - approved_length

percentage_dropped = (dropped_count / original_length) * 100

print(f'Original row count: {original_length}')
print(f'Approved row count: {approved_length}')
print(f'Rows dropped: {dropped_count} ({percentage_dropped:.2f}%)')

## resample turb to daily resolution
df_resampled = df_clean['turb'].resample('D').mean()


# --- 2 --- #
# prepare other daily variables (suspended sediment concentration, discharge)
## import data
daily_param_codes = [
    '00060', # regular Q
    '72137' # tidally-filtered Q
    '80154' # SSC
]

df = nwis.get_record( 
        sites = site,
        service = 'dv',
        start = start_date,
        end = end_date,
    )

daily_param_cols = [
    '00060_Mean',
    '00060_Mean_cd',
    '72137_Mean',
    '72137_Mean_cd',
    '80154_Mean',
    '80154_Mean_cd'
]

df_daily = df[daily_param_cols]

## combine tidally-filtered Q and regular Q, with preference for tidally-filtered
df_daily['72137_Mean'] = df_daily['72137_Mean'].combine_first(
    df_daily['00060_Mean'])
df_daily['72137_Mean_cd'] = df_daily['72137_Mean_cd'].combine_first(df_daily[
    '00060_Mean_cd'])

## remove rows without approved data
df_daily_cleaned = df_daily.loc[df_daily['80154_Mean_cd'] == 'A']
df_daily_cleaned = df_daily.loc[df_daily['72137_Mean_cd'] == 'A']

## verify how how many rows dropped
rows_original = len(df_daily)
rows_approved = len(df_daily_cleaned)

rows_dropped = rows_original - rows_approved
frac_dropped = rows_dropped / rows_original
print(f'Dropped row count: {rows_dropped} ({frac_dropped:.2}%)')

## drop initial columns and rename
df_daily_cleaned.drop(columns = ['00060_Mean', '00060_Mean_cd',
                                 '72137_Mean_cd', '72137_Mean_cd',
                                 '80154_Mean_cd'],
                                 inplace = True)

df_daily_cleaned.rename(columns = {'80154_Mean': 'ssc_mg_L',
                                   '72137_Mean': 'discharge_cfs'},
                                   inplace = True)


# --- 3 --- #
# merge both downsampled iv data and daily data
df_resampled = df_resampled.reset_index()
df_daily_cleaned = df_daily_cleaned.reset_index()
df_all_merged = pd.merge(df_resampled, df_daily_cleaned, 
                         on = 'datetime',
                         how = 'left')