import sqlite3
import pandas as pd
import constants

def get_league_play_by_play_data(season: int) -> pd.DataFrame:
    base_url = 'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_' + str(season) + '.csv.gz'
    pbp_data = pd.read_csv(base_url, compression='gzip', low_memory=False)
    filtered_pbp_data = pbp_data[constants.pbp_filter_list]
    return filtered_pbp_data

def get_player_season_stats(season: int) -> pd.DataFrame:
    base_url = 'https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_season_' + str(season) + '.csv'
    player_stats = pd.read_csv(base_url, low_memory=False)
    return player_stats

if __name__ == "__main__":
    print("Setting up database for custom training data...")
    try:
        db_conn = sqlite3.connect("nfl_llama.db")
        for i in range(2014, 2025):
            pbp_df = get_league_play_by_play_data(i)
            pbp_df.to_sql(f"play_by_play_{i}", db_conn, if_exists="replace")

            player_stats_df = get_player_season_stats(i)
            player_stats_df.to_sql(f"player_stats_{i}", db_conn, if_exists="replace")
    except:
        print("An error occurred while setting up the database.")
    finally:
        db_conn.close()