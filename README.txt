# Global Market Intelligence Dashboard V2.0

## Files
- `app.py`: Streamlit dashboard
- `requirements.txt`: required Python packages

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Run: `pip install -r requirements.txt`
4. Run: `streamlit run app.py`

## Deploy free using Streamlit Community Cloud
1. Create a GitHub repository.
2. Upload `app.py` and `requirements.txt` to the repository root.
3. Sign in at https://share.streamlit.io/ using GitHub.
4. Select the repository, branch `main`, and main file `app.py`.
5. Deploy.

## Notes
- Uses Yahoo Finance through `yfinance`, an unofficial interface.
- Data can be delayed, missing, revised, or unavailable.
- The app refresh button clears Streamlit's 15-minute cache.
- This is an end-of-day research prototype, not a guaranteed real-time data service.
- Verify material price moves against primary exchange or licensed data.
