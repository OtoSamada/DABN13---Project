import pandas as pd
import numpy as np

def clean_data(df):
    # remove duplicates
    #df.drop_duplicates(inplace=True)

    
    df["arrival_date_month"] = df["arrival_date_month"].replace({
        "January": 1, "February": 2, "March":3, "April": 4, "May": 5, "June": 6,
        "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,})
    
    df['country'] = np.where(df['country'].isna(),
                             'Other',
                             df['country'])
    
    conditions = [
    df['country'].isin(['PRT', 'GBR', 'ESP', 'IRL', 'FRA', 'ROU', 'NOR', 'POL', 'DEU', 'BEL', 'CHE', 'GRC', 'ITA', 'NLD', 'DNK', 'RUS', 'SWE', 'EST', 'CZE', 'FIN', 'LUX', 'SVN', 'ALB', 'UKR', 'SMR', 'LVA', 'SRB', 'AUT', 'BLR', 'LTU', 'TUR', 'HRV', 'AND', 'GIB', 'URY', 'JEY', 'GGY', 'SVK', 'HUN', 'BIH', 'BGR', 'CIV', 'MKD', 'ISL', 'MLT', 'IMN', 'LIE', 'MNE', 'FRO']),
    df['country'].isin(['USA', 'CAN', 'MEX', 'PRI', 'JAM', 'CYM', 'ZMB', 'KNA', 'TWN', 'GLP', 'BRB', 'DMA', 'PYF', 'ASM', 'UMI', 'VGB']),
    df['country'].isin(['ARG', 'BRA', 'CHL', 'URY', 'COL', 'VEN', 'SUR', 'ECU', 'PER', 'BOL', 'PRY', 'GUY']),
    df['country'].isin(['CHN', 'IND', 'KOR', 'HKG', 'IRN', 'ARE', 'GEO', 'ARM', 'ISR', 'PHL', 'SEN', 'IDN', 'JPN', 'KWT', 'MDV', 'THA', 'MYS', 'LKA', 'SGP', 'MMR', 'UZB', 'KAZ', 'BDI', 'SAU', 'VNM', 'TJK', 'PAK', 'IRQ', 'NPL', 'BGD', 'QAT', 'JAM', 'MAC', 'TGO', 'RWA', 'KHM', 'SYR', 'JAM', 'JEY']),
    df['country'].isin(['MOZ', 'MAR', 'AGO', 'ZAF', 'EGY', 'NGA', 'KEN', 'ZWE', 'DZA', 'TUN', 'CMR', 'CIV', 'COM', 'UGA', 'GAB', 'GNB', 'MRT', 'DJI', 'STP', 'KEN', 'TZA', 'SDN', 'LBR', 'LKA']),
    df['country'].isin(['AUS', 'FJI', 'NZL', 'PNG', 'PLW', 'NCL', 'KIR', 'FSM', 'SLB', 'NRU', 'TUV', 'TON', 'WSM', 'ASA', 'ATF', 'FJI'])
    ]

    choices = [
        'Europe',
        'North America',
        'South America',
        'Asia',
        'Africa',
        'Oceania'
    ]

    df['continent'] = np.select(conditions, choices, default='Others')
    
    conditions = [
        df['arrival_date_day_of_month'] < 11,
        df['arrival_date_day_of_month'] < 21
    ]

    choices = [
        'BoM', #Beginning of Month
        'MoM', #Middle of Month
    ]

    df['arrival_date_day_of_month']  = np.select(conditions, choices, default='EoM') #End of Month

    columns_to_drop = ["reservation_status", 'reservation_status_date', 'agent', 
                       'company', 'country', 'assigned_room_type', 'arrival_date_year'
                      ]
    df.drop(columns_to_drop, axis=1, inplace=True)
    
    df['adr'] = np.where(
        df['adr'] < 0, 
        df['adr'].median(), 
        df['adr']
    )

    df['children'] = df['children'].fillna(0)
    
    df['adults'] = np.where(df['adults'] <= 0,
                             2, #median of adults
                            df['adults'])
    df['children'] = np.where(df['children'] >= 10,
                             0, #median of children
                            df['children'])
    df['babies'] = np.where(df['babies'] >= 10,
                            0, #median of babies
                            df['babies'])
    df['adr'] = np.where(df['adr'] >= 5000,
                            95, #median of adr
                            df['adr'])
    
    object_features = [col for col in df.columns if pd.api.types.is_object_dtype(df[col])]
    df = pd.get_dummies(df, columns=object_features, drop_first=True)
    
    #columns_to_drop_2 = [
    #                     'arrival_date_day_of_month_MoM'
    #                    ,'meal_SC'
    #                    ,'reserved_room_type_G'
    #                    ,'reserved_room_type_L'
    #                    ,'stays_in_weekend_nights'
    #                    ,'children']
    #df.drop(columns_to_drop_2, axis=1, inplace=True)
    # reset index
    expected_columns = ['is_canceled', 'lead_time', 'arrival_date_month',
             'arrival_date_week_number', 'stays_in_weekend_nights',
             'stays_in_week_nights', 'adults', 'children', 'babies',
             'is_repeated_guest', 'previous_cancellations',
             'previous_bookings_not_canceled', 'booking_changes',
             'days_in_waiting_list', 'adr', 'required_car_parking_spaces',
             'total_of_special_requests', 'hotel_Resort Hotel',
             'arrival_date_day_of_month_EoM', 'arrival_date_day_of_month_MoM',
             'meal_FB', 'meal_HB', 'meal_SC', 'meal_Undefined',
             'market_segment_Complementary', 'market_segment_Corporate',
             'market_segment_Direct', 'market_segment_Groups',
             'market_segment_Offline TA/TO', 'market_segment_Online TA',
             'distribution_channel_Direct', 'distribution_channel_GDS',
             'distribution_channel_TA/TO', 'distribution_channel_Undefined',
             'reserved_room_type_B', 'reserved_room_type_C', 'reserved_room_type_D',
             'reserved_room_type_E', 'reserved_room_type_F', 'reserved_room_type_G',
             'reserved_room_type_H', 'reserved_room_type_L',
             'deposit_type_Non Refund', 'deposit_type_Refundable',
             'customer_type_Group', 'customer_type_Transient',
             'customer_type_Transient-Party', 'continent_Asia', 'continent_Europe',
             'continent_North America', 'continent_Oceania', 'continent_Others',
             'continent_South America']

    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_columns]

    df.reset_index(inplace=True, drop=True)

    return df