import time
from datetime import datetime

import requests


def datetime_to_unix(dt):
    """
    Convert a datetime object to a Unix timestamp.

    Parameters:
        dt (datetime): A datetime object.

    Returns:
        int: Unix timestamp.
    """
    return str(int(dt.timestamp()))


def fetch_all_pages(
    url, headers=None, params=None, key="results", page_param="page", rate_limit=30
):
    """
    Function to handle API pagination and fetch all results while respecting rate limits.

    Parameters:
        url (str): The endpoint URL.
        headers (dict, optional): Headers to include in the request.
        params (dict, optional): Initial URL parameters.
        key (str, optional): Key in the JSON response where data is stored.
        page_param (str, optional): Query parameter name for pagination.
        rate_limit (int, optional): Maximum number of API calls per minute.

    Returns:
        list: Aggregated list of all paginated results.
    """
    all_results = []
    page = 1
    params = params or {}
    delay = 60 / rate_limit  # Calculate delay time per request

    while True:
        params[page_param] = page
        response = make_request("GET", url, headers=headers, params=params)

        if isinstance(response, dict) and key in response:
            results = response[key]
            if not results:
                break
            all_results.extend(results)
            page += 1
            time.sleep(delay)  # Respect rate limit
        else:
            break

    return all_results


def make_request(method, url, headers=None, params=None, data=None, json=None):
    """
    Function to make HTTP requests using the requests library.

    Parameters:
        method (str): HTTP method ('GET', 'POST', 'PUT', 'DELETE').
        url (str): The endpoint URL.
        headers (dict, optional): Headers to include in the request.
        params (dict, optional): URL parameters.
        data (dict, optional): Form data payload.
        json (dict, optional): JSON payload.

    Returns:
        dict: Response JSON if available, otherwise raw text.
    """
    try:
        response = requests.request(
            method, url, headers=headers, params=params, data=data, json=json
        )
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)

        # Try to return JSON, otherwise return raw text
        try:
            return response.json()
        except requests.JSONDecodeError:
            return response.text

    except requests.RequestException as e:
        return {"error": str(e)}
