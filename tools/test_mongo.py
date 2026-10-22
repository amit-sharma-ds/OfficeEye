"""Check the MongoDB connection: connect with MONGO_URI (from .env), insert
one sample document, read it back, and remove it again.

    py -3.12 tools/test_mongo.py

Prints the database name and the result - never the connection string.
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mongo                                        # noqa: E402


def main():
    try:
        db = mongo.get_db()
    except mongo.MongoSetupError as e:
        print("NOT CONNECTED:", e)
        return 1
    print("connected - database:", db.name)
    sample = {"_id": "selftest-{}".format(datetime.now().strftime("%Y%m%d-%H%M%S")),
              "kind": "selftest", "note": "OfficeEye connection test",
              "at": datetime.now().isoformat(timespec="seconds")}
    db.selftest.insert_one(sample)
    print("inserted:", sample["_id"])
    back = db.selftest.find_one({"_id": sample["_id"]})
    print("read back:", {k: back[k] for k in ("_id", "kind", "note", "at")} if back else None)
    ok = back is not None and back["note"] == sample["note"]
    db.selftest.delete_one({"_id": sample["_id"]})       # leave nothing behind
    print("removed the sample again")
    print("TEST PASSED" if ok else "TEST FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
