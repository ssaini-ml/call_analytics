#!/usr/bin/env python3
"""
Exploratory Data Analysis script for Call Analytics - Conversational Intelligence.
Analyzes call data and generates visualizations.
"""

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import pytz
import seaborn as sns
from matplotlib.figure import Figure

from call_analytics_conversational.settings import PATH_AIRCALL_PROCESSED

def parse_args():
    parser = argparse.ArgumentParser(description='Perform exploratory data analysis on Aircall data')
    parser.add_argument('--input-file', type=str, required=True,
                      help='Input CSV file path (relative to PATH_AIRCALL_PROCESSED)')
    parser.add_argument('--output-dir', type=str, default='plots',
                      help='Directory to save plots (default: plots)')
    parser.add_argument('--save-plots', action='store_true',
                      help='Save plots to output directory')
    return parser.parse_args()

def load_and_preprocess_data(input_file: str) -> pd.DataFrame:
    """Load and preprocess the call data"""
    # Construct full path
    full_path = os.path.join(PATH_AIRCALL_PROCESSED, input_file)
    
    # Load data
    calls_df = pd.read_csv(full_path, encoding="latin-1")
    
    # Replace PII artifacts with NA
    calls_df.replace("Error: Document text is empty.", pd.NA, inplace=True)
    
    # Convert timestamp columns to datetime
    for col in ["started_at", "answered_at", "ended_at"]:
        calls_df[col] = calls_df[col].apply(
            lambda x: (
                datetime.fromtimestamp(int(x), tz=timezone.utc).astimezone(
                    pytz.timezone("Europe/Berlin")
                )
                if pd.notna(x)
                else pd.NA
            )
        )
    
    # Add day of week
    calls_df["day_of_week"] = calls_df["started_at"].dt.day_name()
    calls_df.day_of_week = pd.Categorical(
        calls_df.day_of_week,
        categories=[
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday",
        ],
        ordered=True,
    )
    
    # Calculate token estimates
    for ci_feature in ["transcription", "summary", "topics"]:
        calls_df[f"token_estimate_{ci_feature}"] = calls_df[ci_feature].apply(
            lambda x: len(x)/4 if pd.notna(x) else 0
        )
    
    return calls_df

def plot_calls_per_day(calls_df: pd.DataFrame, save_path: Optional[str] = None) -> Figure:
    """Plot number of calls per day"""
    calls_per_day = calls_df.groupby(calls_df["started_at"].dt.date).size()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    calls_per_day.plot(kind='bar', color='skyblue', edgecolor='black', ax=ax)
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Calls")
    ax.set_title("Number of Calls Per Day")
    plt.xticks(rotation=75)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig

def plot_call_duration_histogram(calls_df: pd.DataFrame, save_path: Optional[str] = None) -> Figure:
    """Plot histogram of call durations"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(calls_df['duration'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    
    ax.set_xlabel("Call Duration (seconds)")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of Call Durations")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig

def plot_call_duration_by_day(calls_df: pd.DataFrame, save_path: Optional[str] = None) -> Figure:
    """Plot boxplot of call durations by day of week"""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x=calls_df["day_of_week"], y=calls_df["duration"], 
                palette="muted", fill=False, ax=ax)
    
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Call Duration (seconds)")
    ax.set_title("Call Duration Distribution by Day of Week")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig

def generate_summary_statistics(calls_df: pd.DataFrame) -> Dict:
    """Generate summary statistics for the dataset"""
    # Call volume summary
    volume_summary = calls_df.pivot_table(
        values=["id", "recording", "transcription", "summary", "topics", "sentiment"],
        index=["number_name"],
        aggfunc={
            "id": "count",
            "recording": "count",
            "transcription": "count",
            "summary": "count",
            "topics": "count",
            "sentiment": "count",
        },
        observed=False
    )
    
    # Duration statistics
    duration_stats = calls_df.duration.describe()
    
    # Token estimates
    token_stats = calls_df[[
        "token_estimate_transcription",
        "token_estimate_summary",
        "token_estimate_topics"
    ]].describe()
    
    return {
        "volume_summary": volume_summary,
        "duration_stats": duration_stats,
        "token_stats": token_stats
    }

def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory if saving plots
    if args.save_plots:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess data
    print(f"Loading data from {args.input_file}...")
    calls_df = load_and_preprocess_data(args.input_file)
    
    # Generate and print summary statistics
    print("\nGenerating summary statistics...")
    stats = generate_summary_statistics(calls_df)
    
    print("\nCall Volume Summary:")
    print(stats["volume_summary"])
    
    print("\nCall Duration Statistics:")
    print(stats["duration_stats"])
    
    print("\nToken Estimate Statistics:")
    print(stats["token_stats"])
    
    # Generate plots
    print("\nGenerating plots...")
    if args.save_plots:
        plot_calls_per_day(calls_df, output_dir / "calls_per_day.png")
        plot_call_duration_histogram(calls_df, output_dir / "call_duration_hist.png")
        plot_call_duration_by_day(calls_df, output_dir / "call_duration_by_day.png")
        print(f"Plots saved to {output_dir}")
    else:
        plot_calls_per_day(calls_df)
        plot_call_duration_histogram(calls_df)
        plot_call_duration_by_day(calls_df)
        plt.show()

if __name__ == "__main__":
    main() 