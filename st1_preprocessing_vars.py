# import packages
from dataretrieval import nwis
import pandas as pd
from datetime import timedelta


# input variable preprocessing function
def preprocess_vars(show_gaps = False):
    """
    This function imports, combines, and cleans the following data for USGS site code
    11447650: turbidity (FNU), discharge (cfs), suspended sediment concentration (mg/L).

    Parameters:
        show_gaps (bool, optional): Option to display gaps in imported data. 
        Defaults to False.

    Returns:
        dat(pd.DataFrame): Final dataframe for LSTM training.
    """

    # 1 --- TURBIDITY IMPORTATION
    site = '11447650' # specify USGS site code
    parameter_code = '63680' # turbidity (FNU)
    start_date = '1948-01-01'
    end_date = '2025-08-08'

    ## pull instantaneous values
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
    df_filtered['turb_fnu'] = df_filtered[
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
    df_clean = df_filtered.loc[df_filtered['turb_cd'] == 'A'].copy()
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
    df_resampled = df_clean['turb_fnu'].resample('D').mean()


    # 2 --- Q AND SSC IMPORTATION
    daily_param_codes = [
    '00060', # regular Q
    '72137', # tidally-filtered Q
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

    df_daily = df[daily_param_cols].copy()

    ## combine tidally-filtered Q and regular Q, with preference for tidally-filtered
    df_daily['72137_Mean'] = df_daily['72137_Mean'].combine_first(
        df_daily['00060_Mean'])
    df_daily['72137_Mean_cd'] = df_daily['72137_Mean_cd'].combine_first(df_daily[
        '00060_Mean_cd'])

    ## remove rows without approved data
    df_daily_cleaned = df_daily.loc[
        (df_daily['80154_Mean_cd'] == 'A') & (df_daily['72137_Mean_cd'] == 'A')].copy()

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
                                    '72137_Mean': 'Q_cfs'},
                                    inplace = True)


    # 3 --- MERGE IV AND DV DATA
    df_resampled = df_resampled.reset_index()
    df_daily_cleaned = df_daily_cleaned.reset_index()

    df_all_merged = pd.merge(df_resampled, df_daily_cleaned, 
                            on = 'datetime',
                            how = 'left')
    

    # 4 --- EXTRACT AND QUANTIFY DATA GAPS
    if show_gaps is True:
        ## identify percentage of NA values per variable
        nan_turb_count = df_all_merged['turb_fnu'].isna().sum()
        nan_ssc_count = df_all_merged['ssc_mg_L'].isna().sum()
        nan_Q_count = df_all_merged['Q_cfs'].isna().sum()

        nan_turb_frac = (nan_turb_count / len(df_all_merged['turb_fnu'])) * 100
        nan_ssc_frac = (nan_ssc_count / len(df_all_merged['ssc_mg_L'])) * 100
        nan_Q_frac = (nan_Q_count / len(df_all_merged['Q_cfs'])) * 100

        print(f'Missing turb_fnu rows: {nan_turb_count} ({nan_turb_frac}%)')
        print(f'Missing ssc_mg_L rows: {nan_ssc_count} ({nan_ssc_frac}%)')
        print(f'Missing Q_cfs rows: {nan_Q_count} ({nan_Q_frac}%)')

        ## identify length of gaps in each variable
        from datetime import timedelta

        ### use diff() to extract series of differences between datetime row n and n - 1
        deltas = df_all_merged['datetime'].diff()[1:]
        gaps = deltas[deltas > timedelta(days = 1)]
        print(f'{len(gaps)} datetime gaps with average gap duration: {gaps.mean()}')

        ## extract gap lengths and associated date ranges
        var_columns = [
            'turb_fnu',
            'ssc_mg_L',
            'Q_cfs'
        ]

        for var in var_columns:
            gap_sizes = []
            gap_dates = []
            gap_size = 0
            gap_startdate = None
            
            print(f'--- {var} ---')
            for index, row in df_all_merged.iterrows():
                if pd.isna(row[var]):
                    if gap_size == 0:
                        gap_startdate = row['datetime']
                    gap_size += 1
                else:
                    if gap_size > 0:
                        gap_sizes.append(gap_size)
                        gap_dates.append((gap_startdate, row['datetime']))
                        gap_size = 0
                        gap_startdate = None
            
            ### case where gap size > 0 but end of df is reached
            if gap_size > 0:
                gap_sizes.append(gap_size)
                gap_dates.append((gap_startdate, df_all_merged['datetime'].iloc[-1]))

            sorted_gaps = sorted(zip(gap_sizes, gap_dates), reverse = True)    

            ### print gap information for given variable
            print(f'Top 10 gaps in {var} column:')
            for size, dates in sorted_gaps[:10]:
                print(f'Length: {size}, Date Range:{dates}')
            print(f'Total number of gaps: {len(sorted_gaps)}')


    # 5 --- FILL REMAINING GAPS
    ## set cutoff date based on turbidity data
    df_all_merged = df_all_merged.loc[df_all_merged['datetime'] <= '2023-09-30'].copy()

    ## interpolate missing values
    df_filled = df_all_merged.interpolate(method = 'linear')

    ## verify that NaNs are filled
    print(f'Original: {df_all_merged.isna().sum()}')
    print(f'Filled: {df_filled.isna().sum()}')


    # 6 -- VERIFY CHANGES 
    ## compare df lengths
    print(f'Original length: {len(df_all_merged)}')
    print(f'After filling: {len(df_filled)}')

    ## verify that NaNs are filled
    print(f'Original null count: {df_all_merged.isna().sum()}')
    print(f'Filled null count: {df_filled.isna().sum()}')

    ## verify stats
    print('Original data: ')
    print(df_all_merged.describe())

    print('Filled data: ')
    print(df_filled.describe())



    dat = df_filled.copy()

    return dat
