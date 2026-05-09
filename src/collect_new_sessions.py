import pandas as pd
import os


def store_session(session_data):

    file_path = "data/live_sessions.csv"

    df = pd.DataFrame([session_data])

    if os.path.exists(file_path):

        df.to_csv(
            file_path,
            mode='a',
            header=False,
            index=False
        )

    else:

        df.to_csv(file_path, index=False)

    print("Trusted session stored")