import os
import time

import boto3
from boto3.dynamodb.conditions import Key

_table = None


def _t():
    global _table
    if _table is None:
        db = boto3.resource("dynamodb", region_name=os.getenv("AWS_REGION", "ap-northeast-2"))
        _table = db.Table(os.getenv("DDB_TABLE", "edu_tutor_answers"))
    return _table


def add_answer(student_id, qid, response):
    _t().put_item(Item={"student_id": student_id, "ts": int(time.time() * 1000),
                        "qid": int(qid), "response": int(response)})


def get_history(student_id, limit=200):
    r = _t().query(KeyConditionExpression=Key("student_id").eq(student_id),
                   ScanIndexForward=False, Limit=limit)
    return [{"qid": int(i["qid"]), "response": int(i["response"])}
            for i in reversed(r["Items"])]