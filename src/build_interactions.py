from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SEQ_CSV = ROOT / "data" / "raw" / "question_level" / "train_valid_sequences_quelevel.csv"
OUT = ROOT / "data" / "processed" / "interactions.parquet"

COLS = ["questions", "concepts", "responses", "timestamps", "selectmasks"]
CHUNK = 3000


def explode_chunk(chunk):
    tmp = chunk.copy()
    tmp["seq_id"] = tmp.index
    for c in COLS:
        tmp[c] = tmp[c].astype(str).str.strip(",").str.split(",")
    tmp["pos"] = tmp["questions"].map(lambda x: list(range(len(x))))

    long = tmp.explode(COLS + ["pos"], ignore_index=True)
    long = long[long["questions"] != "-1"].copy()

    long["qid"] = long["questions"].astype("int32")
    long["response"] = long["responses"].astype("int8")
    long["timestamp"] = long["timestamps"].astype("int64")
    long["pos"] = long["pos"].astype("int16")
    long["seq_id"] = long["seq_id"].astype("int32")
    long["uid"] = long["uid"].astype("int64")
    long["fold"] = long["fold"].astype("int8")

    return long[["seq_id", "uid", "fold", "pos", "qid", "concepts", "response", "timestamp"]]


def main():
    seq = pd.read_csv(SEQ_CSV)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    total = 0
    for start in range(0, len(seq), CHUNK):
        part = explode_chunk(seq.iloc[start:start + CHUNK])
        table = pa.Table.from_pandas(part, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(OUT, table.schema)
        writer.write_table(table)
        total += len(part)
        print(f"{start + len(part)}행 처리, 누적 {total:,}건")

    writer.close()
    print(f"완료: {total:,}건 -> {OUT}")


if __name__ == "__main__":
    main()