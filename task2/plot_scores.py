# task2/plot_scores.py
import pandas as pd
import matplotlib.pyplot as plt
import ast
import os

def plot_score_distribution(dataset_path, output_path):
    df = pd.read_csv(dataset_path, index_col=0)
    df = df[df["Action_Type"] == "END_MATCH"]
    names = ast.literal_eval(df.iloc[0]["Match_Score"])
    scores = df["Game_Score"].apply(ast.literal_eval).tolist()
    scores_arr = pd.DataFrame(scores, columns=names)

    plt.figure()
    for n in names:
        plt.plot(scores_arr[n], label=n)
    plt.xlabel("Match")
    plt.ylabel("Score")
    plt.title("Score Progression")
    plt.legend()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    folder = "outputs"
    dataset_file = os.path.join(folder, os.listdir(folder)[0], "dataset", "game_dataset.pkl.csv")
    plot_score_distribution(dataset_file, os.path.join(folder, "score_progression.png"))