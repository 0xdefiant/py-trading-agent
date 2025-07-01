import os
import csv

TOKEN_DATA_DIR = os.path.join(os.path.dirname(__file__), 'tokenData')

# Helper to detect and fix unix timestamp
def fix_unix_timestamp(ts):
    ts_str = str(ts)
    # If it's longer than 10 digits and ends with zeros, trim to 10 digits
    if len(ts_str) > 10 and ts_str.endswith('000'):
        return ts_str[:10]
    return ts_str

# Helper to format numbers to 6 decimal places (except unix column)
def format_number(val):
    try:
        # Don't format if it's empty or not a number
        if val == '' or val is None:
            return val
        # Try to convert to float
        float_val = float(val)
        # If it's an integer, format as 6 decimal places
        return f'{float_val:.6f}'
    except ValueError:
        return val

def clean_csv_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # Remove first line if it contains CryptoDataDownload.com
    if lines and 'CryptoDataDownload.com' in lines[0]:
        lines = lines[1:]
    # Use csv reader to process header and rows
    reader = csv.reader(lines)
    rows = list(reader)
    if not rows:
        return
    header = rows[0]
    # Find the unix column index
    try:
        unix_idx = header.index('unix')
    except ValueError:
        unix_idx = 0  # fallback: assume first column
    # Fix unix column and format numbers in each row
    for i in range(1, len(rows)):
        if len(rows[i]) > unix_idx:
            # Fix unix timestamp
            rows[i][unix_idx] = fix_unix_timestamp(rows[i][unix_idx])
        # Format all other numeric columns to 6 decimal places
        for j in range(len(rows[i])):
            if j != unix_idx:
                rows[i][j] = format_number(rows[i][j])
    # Write back to file
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def process_all_csvs():
    for root, dirs, files in os.walk(TOKEN_DATA_DIR):
        for file in files:
            if file.endswith('.csv'):
                filepath = os.path.join(root, file)
                print(f'Cleaning {filepath}')
                clean_csv_file(filepath)

if __name__ == '__main__':
    process_all_csvs() 