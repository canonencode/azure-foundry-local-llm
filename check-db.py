# Week 3 - Debug script: dump row count and contents of knowledge.db
# .\venv\Scripts\Activate.ps1

import sys
import sqlite3

# See main.py for why this is needed - non-Latin document content would
# otherwise crash on print() with UnicodeEncodeError on Windows' default
# console codepage.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    conn = sqlite3.connect("knowledge.db")
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        count = cursor.fetchone()
        print(f"Total rows in documents table: {count[0]}")

        cursor.execute("SELECT doc_index, content FROM documents ORDER BY doc_index")
        rows = cursor.fetchall()
        for row in rows:
            print(row)
    except sqlite3.OperationalError as exc:
        print(f"Could not read knowledge.db: {exc}")
        print("Run 'python ingest.py' first to create and populate it.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
