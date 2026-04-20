import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
INPUT_FILE = r"D:\TEST OUPUTS\results\intensity_mapped_results.csv"
OUTPUT_REPORT = r"D:\TEST OUPUTS\results\final_report.txt"
# ---------------------

def generate_report():
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("Error: File not found. Check the path.")
        return

    # Open the text file for writing
    with open(OUTPUT_REPORT, "w") as f:
        
        # We use 'file=f' to tell print() to write to the file instead of the screen
        print("--- JUDO MATCH ANALYSIS REPORT ---\n", file=f)

        # 1. TOTAL DURATION PER PHASE
        phase_counts = df['phase'].value_counts()
        total_rows = len(df)
        
        print("1. Phase Dominance (Total Duration):", file=f)
        for phase, count in phase_counts.items():
            percent = (count / total_rows) * 100
            print(f"   - {phase}: {percent:.1f}%", file=f)

        # 2. TRANSITION ANALYSIS
        df['prev_phase'] = df['phase'].shift(1)
        transitions = df[df['phase'] != df['prev_phase']]
        num_transitions = len(transitions) - 1
        
        print(f"\n2. Match Flow:", file=f)
        print(f"   - Total Phase Transitions: {num_transitions}", file=f)
        
        if num_transitions > 0:
            avg_rows_between = total_rows / num_transitions
            print(f"   - Avg. Time Between Transitions: {avg_rows_between:.0f} seconds", file=f)

    print(f"Report saved to: {OUTPUT_REPORT}")

if __name__ == "__main__":
    generate_report()