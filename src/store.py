import sqlite3
import time
from contextlib import closing
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "data" / "student.db"


def _conn():
    c = sqlite3.connect(DB)
    c.execute("create table if not exists answers "
              "(student_id text, qid integer, response integer, ts real)")
    return c


def add_answer(student_id, qid, response):
    with closing(_conn()) as c, c:
        c.execute("insert into answers values (?,?,?,?)",
                  (student_id, qid, response, time.time()))


def get_history(student_id, limit=200):
    with closing(_conn()) as c:
        rows = c.execute(
            "select qid, response from answers where student_id=? "
            "order by ts desc limit ?", (student_id, limit)).fetchall()
    return [{"qid": q, "response": r} for q, r in reversed(rows)]

import os

if os.getenv("STORE_BACKEND") == "dynamo":
    from src.store_dynamo import add_answer, get_history  # noqa: F811