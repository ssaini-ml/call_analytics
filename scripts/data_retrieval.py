#!/usr/bin/env python3
"""
Data retrieval script for Call Analytics - Conversational Intelligence.
Retrieves and processes call data from Aircall API.
"""

import argparse
import json
import os
import shutil
import time
from datetime import date, datetime, timedelta
from typing import List, Optional

import nest_asyncio
import pandas as pd
import pytz
from dotenv import load_dotenv
from tqdm import tqdm

from call_analytics_conversational.api_utils import datetime_to_unix, fetch_all_pages, make_request
from call_analytics_conversational.conversation import (
    get_call_summary,
    get_call_topics,
    get_call_transcription,
    get_sentiment_analysis,
)
from call_analytics_conversational.pii_utility import authenticate_pii_client, redact_pii_with_batches
from call_analytics_conversational.settings import (
    AIRCALL_API_BASE64_CREDENTIALS,
    AIRCALL_API_RATELIMIT,
    AIRCALL_API_URL,
    AIRCALL_NUMBERS_CS,
    AIRCALL_NUMBERS_PS,
    PATH_AIRCALL_CALLS,
    PATH_AIRCALL_DATA,
    PATH_AIRCALL_PROCESSED,
    PATH_AIRCALL_SENTIMENTS,
    PATH_AIRCALL_SUMMARIES,
    PATH_AIRCALL_TOPICS,
    PATH_AIRCALL_TRANSCRIPTIONS,
    RANDOM_STATE,
)
from call_analytics_conversational.utils import day_month_list

def parse_args():
    parser = argparse.ArgumentParser(description='Retrieve and process Aircall data')
    parser.add_argument('--business', type=str, choices=['CS', 'PS', 'OTHER'], required=True,
                      help='Business unit to analyze (CS: Customer Service, PS: Pharma Service, OTHER: Custom numbers)')
    parser.add_argument('--start-date', type=str, required=True,
                      help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end-date', type=str, required=True,
                      help='End date in YYYY-MM-DD format')
    parser.add_argument('--sample-size', type=int, default=2,
                      help='Number of calls to download Conversational Intelligence features for')
    parser.add_argument('--custom-numbers', type=str, nargs='+',
                      help='Custom phone numbers to analyze (required if business is OTHER)')
    parser.add_argument('--ci-features', type=str, nargs='+',
                      default=["sentiments", "summary", "topics", "transcription"],
                      help='Conversational Intelligence features to retrieve')
    return parser.parse_args()

def setup_directories(ci_features: List[str]):
    """Create necessary directories for data storage"""
    os.makedirs(PATH_AIRCALL_CALLS, exist_ok=True)
    os.makedirs(PATH_AIRCALL_PROCESSED, exist_ok=True)
    for ci_feature in ci_features:
        os.makedirs(f'{PATH_AIRCALL_DATA}/{ci_feature}', exist_ok=True)

def get_phone_numbers(business: str, custom_numbers: Optional[List[str]] = None) -> List[str]:
    """Get phone numbers based on business unit"""
    if business == 'CS':
        return AIRCALL_NUMBERS_CS
    elif business == 'PS':
        return AIRCALL_NUMBERS_PS
    elif business == 'OTHER':
        if not custom_numbers:
            raise ValueError('Custom numbers must be provided when business is OTHER')
        return custom_numbers
    else:
        raise ValueError('Invalid business')

def retrieve_calls(numbers: List[str], start_date: date, end_date: date) -> pd.DataFrame:
    """Retrieve calls from Aircall API for specified numbers and date range"""
    day_months = day_month_list(start_date, end_date)
    all_calls = []

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

        if calls_df.empty:
            continue

        # Process and save data
        process_and_save_calls(calls_df, number, start_date, end_date)

    return pd.concat([pd.read_csv(f"{PATH_AIRCALL_CALLS}/{file}") 
                     for file in os.listdir(PATH_AIRCALL_CALLS)
                     if file.startswith(f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}")
                     and file.endswith(tuple([f"{number.replace('+', '').replace(' ', '')}.csv" 
                                            for number in numbers]))])

def process_and_save_calls(calls_df: pd.DataFrame, number: str, start_date: date, end_date: date):
    """Process and save call data to CSV"""
    # Unpack nested data
    calls_df["number_id"] = calls_df["number"].apply(lambda x: x["id"])
    calls_df["number_digits"] = calls_df["number"].apply(lambda x: x["digits"])
    calls_df["number_name"] = calls_df["number"].apply(lambda x: x["name"])
    calls_df["number_country"] = calls_df["number"].apply(lambda x: x["country"])

    # Filter relevant columns
    calls_df = calls_df[[
        "id", "sid", "direction", "status", "missed_call_reason",
        "started_at", "answered_at", "ended_at", "duration",
        "recording", "number_id", "number_digits", "number_name",
        "number_country", "country_code_a2"
    ]]

    # Save to CSV
    output_file = f"{PATH_AIRCALL_CALLS}/{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_calls_{number.replace('+', '').replace(' ', '')}.csv"
    calls_df.to_csv(output_file, index=False)
    print(f"Saved calls data to {output_file}")

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Convert date strings to date objects
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()
    
    # Setup directories
    setup_directories(args.ci_features)
    
    # Get phone numbers
    numbers = get_phone_numbers(args.business, args.custom_numbers)
    print(f'Fetching calls for {args.business} from {start_date} to {end_date}.\nThis includes the following Numbers:\n{numbers}')
    
    # Retrieve and process calls
    calls_df = retrieve_calls(numbers, start_date, end_date)
    
    # Print summary
    summary = calls_df.pivot_table(
        values=["id", "recording"],
        index=["number_name", "direction"],
        aggfunc={"id": "count", "recording": "count"},
        observed=False,
        margins=True,
        margins_name="Total",
    )
    print("\nCall Summary:")
    print(summary)

if __name__ == "__main__":
    nest_asyncio.apply()  # Required for running async code in Jupyter-like environment
    main() 