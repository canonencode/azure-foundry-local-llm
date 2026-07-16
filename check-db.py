import sqlite3

conn = sqlite3.connect("knowledge.db")
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM documents")
count = cursor.fetchone()
print(f"Total rows in documents table: {count[0]}")

cursor.execute("SELECT doc_index, content FROM documents ORDER BY doc_index")
rows = cursor.fetchall()
for row in rows:
    print(row)

conn.close()