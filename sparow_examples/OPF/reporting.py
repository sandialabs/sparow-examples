import json
import pandas as pd
import matplotlib.pyplot as plt

def print_summary(title, summary):
    print(f"\n{title}")
    print("-" * 60)
    for method, result in summary.items():
        print(f"{method}:")
        print(f"  CPU Time: {result['cpu_time']:.4f} seconds")
        print(f"  Objective Value: {result['objective']:.4f}")
    print("-" * 60)

def save_summary(summary, filename):
    with open(filename, "w") as f:
        json.dump(summary, f, indent=2)


def plot_opf_runs(ef_summary, ph_summary, reevaluated_summary, filename="OPF_runs.png"):
    hf_reference = reevaluated_summary["HF_EF"]["objective"]

    plot_rows = []

    # EF methods
    for method, result in ef_summary.items():
        cpu_time = result["cpu_time"]
        hf_obj = reevaluated_summary[method]["objective"]
        percent_error = ((hf_obj - hf_reference) / hf_reference) * 100

        plot_rows.append({
            "Method": method,
            "CPU Seconds": cpu_time,
            "Percent Error": percent_error,
        })

    # PH methods
    for method, result in ph_summary.items():
        cpu_time = result["cpu_time"]
        hf_obj = reevaluated_summary[method]["objective"]
        percent_error = ((hf_obj - hf_reference) / hf_reference) * 100

        plot_rows.append({
            "Method": method,
            "CPU Seconds": cpu_time,
            "Percent Error": percent_error,
        })

    plot_df = pd.DataFrame(plot_rows)

    marker_map = {
        "HF_EF": "x",
        "HF_PH": "x",
        "LF_EF": "*",
        "LF_PH": "*",
        "MF_EF": "o",
        "MF_PH_Random": "o",
        "MF_PH_Similar": "o",
        "MF_PH_Dissimilar": "o",
    }

    color_map = {
        "HF_EF": "black",
        "HF_PH": "blue",
        "LF_EF": "black",
        "LF_PH": "blue",
        "MF_EF": "black",
        "MF_PH_Random": "gold",
        "MF_PH_Similar": "green",
        "MF_PH_Dissimilar": "red",
    }

    plt.figure(figsize=(10, 6))

    for _, row in plot_df.iterrows():
        method = row["Method"]
        plt.scatter(
            row["CPU Seconds"],
            row["Percent Error"],
            marker=marker_map.get(method, "o"),
            color=color_map.get(method, "black"),
            label=method,
            s=100,
        )

    plt.xlabel("CPU Time (seconds)", fontsize=15)
    plt.ylabel("Percent Error (%)", fontsize=15)
    plt.axhline(0, color="grey", lw=0.8, ls="--")
    plt.grid(True)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), title="Methods", fontsize=13)
    plt.yticks(fontsize=13)
    plt.xticks(fontsize=13)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()