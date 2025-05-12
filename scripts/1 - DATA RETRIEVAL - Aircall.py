#!/usr/bin/env python
# coding: utf-8

import json
import os
import shutil
import time
from datetime import date, datetime

import nest_asyncio
import pandas as pd
import pytz
from tqdm import tqdm

from call_analytics.api_utils import datetime_to_unix, fetch_all_pages, make_request
from call_analytics.conversation import Conversation
from call_analytics.pii_utility import authenticate_pii_client, redact_pii_with_batches
from call_analytics.settings import (
    AIRCALL_API_BASE64_CREDENTIALS,
    AIRCALL_API_RATELIMIT,
    AIRCALL_API_URL,
    AIRCALL_NUMBERS_CS,
    AIRCALL_NUMBERS_PS,
    PATH_AIRCALL_CALLS,
    PATH_AIRCALL_DATA,
    PATH_AIRCALL_PROCESSED,
    RANDOM_STATE,
)
from call_analytics.utils import day_month_list

def retrieve_calls(business='OTHER', start_date=None, end_date=None, sample_size=2):
    """
    Retrieve calls from Aircall API for the specified business and date range.
    
    Args:
        business (str): Business type ('CS', 'PS', or 'OTHER')
        start_date (date): Start date for call retrieval
        end_date (date): End date for call retrieval
        sample_size (int): Number of calls to download Conversational Intelligence features for
    """
    # Set default dates if not provided
    if start_date is None:
        start_date = date(2025, 3, 8)
    if end_date is None:
        end_date = date(2025, 3, 9)
        
    # Set business-specific parameters
    if business == 'CS':
        numbers = AIRCALL_NUMBERS_CS
    elif business == 'PS':
        numbers = AIRCALL_NUMBERS_PS
    elif business == 'OTHER':
        numbers = ['+31 85 888 1579', '+31 85 888 1529']
    else:
        raise ValueError('Invalid business')
    
    # Create necessary directories
    os.makedirs(PATH_AIRCALL_CALLS, exist_ok=True)
    os.makedirs(PATH_AIRCALL_PROCESSED, exist_ok=True)
    for feature in ["sentiments", "summary", "topics", "transcription"]:
        os.makedirs(f'{PATH_AIRCALL_DATA}/{feature}', exist_ok=True)
    
    print(f'Fetching calls for {business} from {start_date} to {end_date}.')
    print(f'This includes the following Numbers:\n{numbers}')
    
    # Generate list of days and months in the specified range
    day_months = day_month_list(start_date, end_date)
    
    # List to store all calls
    all_calls = []
    
    # Per number, fetch all calls for each day in the specified range
    for number in numbers:
        for day, month in day_months:
            calls = fetch_all_pages(
                url=f"{AIRCALL_API_URL}/calls/search",
                headers={"Authorization": f"Basic {AIRCALL_API_BASE64_CREDENTIALS}"},
                params={
                    "from": datetime_to_unix(datetime(2025, month, day, 0, 0, 0, tzinfo=pytz.timezone("Europe/Berlin"))),
                    "to": datetime_to_unix(datetime(2025, month, day, 23, 59, 59, tzinfo=pytz.timezone("Europe/Berlin"))),
                    "direction": "inbound",
                    "phone_number": number,
                },
                key="calls",
                page_param="page",
                rate_limit=AIRCALL_API_RATELIMIT,
            )
            all_calls.extend(calls)
        
        calls_df = pd.DataFrame(all_calls)
        print(f"Retrieved {calls_df.shape[0]} calls for {number}.")
        
        if not calls_df.empty:
            # Unpack relevant nested data
            calls_df["number_id"] = calls_df["number"].apply(lambda x: x["id"])
            calls_df["number_digits"] = calls_df["number"].apply(lambda x: x["digits"])
            calls_df["number_name"] = calls_df["number"].apply(lambda x: x["name"])
            calls_df["number_country"] = calls_df["number"].apply(lambda x: x["country"])
            
            # Filter relevant columns
            calls_df = calls_df[[
                "id", "sid", "direction", "status", "missed_call_reason",
                "started_at", "answered_at", "ended_at", "duration", "recording",
                "number_id", "number_digits", "number_name", "number_country",
                "country_code_a2",
            ]]
            
            # Save to CSV
            output_file = f"{PATH_AIRCALL_CALLS}/{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_calls_{number.replace('+', '').replace(' ', '')}.csv"
            calls_df.to_csv(output_file, index=False)
            print(f"Saved calls to {output_file}")
    
    return load_calls(start_date, end_date, numbers)

def load_calls(start_date, end_date, numbers):
    """Load calls from saved CSV files."""
    calls_df = pd.DataFrame()
    
    for file in os.listdir(PATH_AIRCALL_CALLS):
        if (file.startswith(f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}") and 
            file.endswith(tuple([f"{number.replace('+', '').replace(' ', '')}.csv" for number in numbers]))):
            calls_df = pd.concat([calls_df, pd.read_csv(f"{PATH_AIRCALL_CALLS}/{file}")])
    
    if not calls_df.empty:
        print("\nCall Summary:")
        summary = calls_df.pivot_table(
            values=["id", "recording"],
            index=["number_name", "direction"],
            aggfunc={"id": "count", "recording": "count"},
            observed=False,
            margins=True,
            margins_name="Total",
        )
        print(summary)
    
    return calls_df

def main():
    """Main function to run the script."""
    try:
        # You can modify these parameters as needed
        business = 'OTHER'
        start_date = date(2025, 3, 8)
        end_date = date(2025, 3, 9)
        sample_size = 2
        
        calls_df = retrieve_calls(business, start_date, end_date, sample_size)
        
        if calls_df.empty:
            print("No calls were retrieved for the specified parameters.")
        else:
            print(f"\nTotal calls retrieved: {len(calls_df)}")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()

