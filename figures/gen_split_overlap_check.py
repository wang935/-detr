from pathlib import Path
import csv


ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "train_posonly": ROOT / "data" / "dfire_local" / "train_posonly.txt",
    "train_equalstep": ROOT / "data" / "dfire_local" / "train_posonly_equalstep.txt",
    "train_full": ROOT / "data" / "dfire_local" / "train_full.txt",
    "train_sched": ROOT / "data" / "stage5_pv_v2" / "dfire" / "train_hardneg_sched.txt",
    "test": ROOT / "data" / "dfire_local" / "test.txt",
}


def read_set(path):
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip()
    }


sets = {name: read_set(path) for name, path in FILES.items() if path.exists()}
rows = []
for train_name in ["train_posonly", "train_equalstep", "train_full", "train_sched"]:
    overlap = sets[train_name] & sets["test"]
    rows.append(
        {
            "train_split": train_name,
            "train_unique": len(sets[train_name]),
            "test_unique": len(sets["test"]),
            "overlap_with_test": len(overlap),
        }
    )

hardneg_only = sets["train_full"] - sets["train_posonly"]
rows.append(
    {
        "train_split": "train_full_minus_posonly",
        "train_unique": len(hardneg_only),
        "test_unique": len(sets["test"]),
        "overlap_with_test": len(hardneg_only & sets["test"]),
    }
)

out = ROOT / "figures" / "paper_split_overlap_check.csv"
with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["train_split", "train_unique", "test_unique", "overlap_with_test"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved: {out}")

