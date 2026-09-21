"""Turn b-mc2/sql-create-context rows into prompt/completion chat pairs.

Each row has a question, the CREATE TABLE statements it refers to, and the SQL
answer. The model sees the schema and question and learns to emit only SQL.
"""
from datasets import Dataset, load_dataset

PROMPT = (
    "Given this schema:\n{context}\n\n"
    "Write one SQLite query that answers: {question}\n"
    "Return only the SQL."
)


def to_pair(row: dict) -> dict:
    return {
        "prompt": [{"role": "user", "content": PROMPT.format(context=row["context"], question=row["question"])}],
        "completion": [{"role": "assistant", "content": row["answer"]}],
    }


def load_splits(dataset_id: str, n_train: int, n_eval: int, seed: int) -> tuple[Dataset, Dataset]:
    """Shuffle once, then take disjoint train and eval slices."""
    ds = load_dataset(dataset_id, split="train").shuffle(seed=seed)
    train = ds.select(range(n_train)).map(to_pair, remove_columns=ds.column_names)
    eval_ = ds.select(range(n_train, n_train + n_eval)).map(to_pair, remove_columns=ds.column_names)
    return train, eval_
