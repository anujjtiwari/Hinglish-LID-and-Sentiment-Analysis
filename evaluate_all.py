from pathlib import Path
import subprocess
import sys
import pandas as pd


GROUND_FILE = Path("ground.csv")


def main():

    print("=" * 80)
    print("COMI-LINGUA LID - BATCH EVALUATION")
    print("=" * 80)

    # Check ground file
    if not GROUND_FILE.exists():
        print(f"\nERROR: {GROUND_FILE} not found.")
        return

    # Automatically find prediction CSVs
    prediction_files = sorted(
        p for p in Path(".").glob("*.csv")
        if p.name != GROUND_FILE.name
        and "evaluation" not in p.name.lower()
    )

    print(f"\nGround file: {GROUND_FILE}")
    print(f"Prediction files found: {len(prediction_files)}")

    for p in prediction_files:
        print(f"  - {p.name}")

    if not prediction_files:
        print("\nERROR: No prediction CSV files found.")
        return

    results = []

    # Evaluate every prediction file
    for prediction_file in prediction_files:

        print("\n")
        print("#" * 80)
        print(f"EVALUATING: {prediction_file.name}")
        print("#" * 80)

        command = [
            sys.executable,
            "eval.py",
            str(GROUND_FILE),
            str(prediction_file)
        ]

        process = subprocess.run(
            command,
            capture_output=True,
            text=True
        )

        # Show evaluator output
        print(process.stdout)

        if process.returncode != 0:
            print("ERROR:")
            print(process.stderr)
            continue

        # Extract macro metrics
        macro_precision = None
        macro_recall = None
        macro_f1 = None

        for line in process.stdout.splitlines():

            line = line.strip()

            if line.startswith("Macro Precision"):
                macro_precision = float(
                    line.split(":")[-1].strip()
                )

            elif line.startswith("Macro Recall"):
                macro_recall = float(
                    line.split(":")[-1].strip()
                )

            elif line.startswith("Macro F1"):
                macro_f1 = float(
                    line.split(":")[-1].strip()
                )

        results.append({
            "model": prediction_file.stem,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1
        })

    # Nothing evaluated
    if not results:
        print("\nNo evaluation results were generated.")
        return

    # Create results table
    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "macro_f1",
        ascending=False
    )

    # Save results
    output_file = "evaluation_results.csv"

    results_df.to_csv(
        output_file,
        index=False
    )

    print("\n\n")
    print("=" * 80)
    print("FINAL MODEL COMPARISON")
    print("=" * 80)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\n" + "=" * 80)
    print(f"Results saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()