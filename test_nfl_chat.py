import sqlite3
import pandas as pd
import ollama
import chromadb

def get_combined_player_stats(db_conn: sqlite3.Connection, start_year: int, end_year: int) -> pd.DataFrame:
    player_stats_list = []
    for i in range(start_year, end_year + 1):
        player_stats = pd.read_sql(f"SELECT * FROM player_stats_{i} WHERE season_type='REG'", db_conn)
        player_stats['combined'] = player_stats.apply(lambda row: '; '.join([f"{col}: {row[col]}" for col in player_stats.columns]), axis=1)
        player_stats['combined'] = player_stats['combined'].apply(lambda x: f"{{{x}}}")
        player_stats_list.append(player_stats)
    return pd.concat(player_stats_list)

def embed_player_stats(collection: chromadb.Collection, stats_df: pd.DataFrame):
    for index, row in stats_df.iterrows():
        embed_response = ollama.embeddings(model="mxbai-embed-large", prompt=row["combined"])
        embedding = embed_response["embedding"]
        collection.add(
            ids = [ str(index) + row["player_id"] + str(row["season"]) ],
            embeddings = [embedding],
            metadatas = {
                    "player_short_name": row["player_name"],
                    "player_name": row["player_display_name"],
                    "position": row["position"],
                    "season": row["season"],
                    "team": row["recent_team"] 
                },
            documents = [ row["combined"] ]
        )

def get_combined_pbp() -> pd.DataFrame:
    return

def setup_backend(db_conn: sqlite3.Connection, vector_db_client: chromadb.ClientAPI) -> tuple[chromadb.Collection, chromadb.Collection]:
    print("Setting up data to be embedded and stored in ChromaDB...")
    pbp_collection = vector_db_client.create_collection(name="pbp")

    player_stats_collection = vector_db_client.create_collection(name="player_stats")
    combined_stats = get_combined_player_stats(db_conn, 2024, 2024)
    embed_player_stats(player_stats_collection, combined_stats)

    return player_stats_collection, pbp_collection

def create_data_string(vector_query_result: chromadb.QueryResult) -> str:
    data_str = ""
    print(vector_query_result)
    for doc in vector_query_result["documents"]:
        data_str += f"{doc}\n"
    return data_str
        
def run_chatbot(collection: chromadb.Collection) -> None:
    while True:
        prompt = input("Enter a prompt: ")

        response = ollama.embeddings(model="mxbai-embed-large", prompt=prompt)
        results = collection.query(
            query_embeddings=[response["embedding"]],
            n_results=10
        )
        data = create_data_string(results)

        output = ollama.generate(
            model="llama3.2",
            prompt=f"Using this data: {data}. Respond to this prompt: {prompt}. Be intelligent " +
                    "about the data that is given. Based on the prompt you may not need to look " + 
                    "at everything. Take that into account."
        )

        print("\n")
        print(f"Prompt: {prompt}")
        print("\n")
        print(f"Data: {data}")
        print("\n")
        print(output['response'])

if __name__ == "__main__":
    print("Running prototype NFL chatbot!")
    try:
        db_conn = sqlite3.connect("nfl_llama.db")
        vector_client = chromadb.Client()
        print("Connected to ChromaDB.")
        print(vector_client.list_collections())
        player_stats_collection, pbp_collection = setup_backend(db_conn, vector_client)
        run_chatbot(player_stats_collection)
    finally:
        db_conn.close()


        # Give me a rundown of the Drake London's 2024 season