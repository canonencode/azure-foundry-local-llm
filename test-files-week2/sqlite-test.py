#Week 2 - SQLite Sandbox
# .\venv\Scripts\Activate.ps1

import sqlite3

def main():

    #Connect to a database
    conn = sqlite3.connect("test.db")
    cursor = conn.cursor()

    #Create a table
    cursor.execute("DROP TABLE IF EXISTS documents")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            content TEXT,
            embedding TEXT
        )
    """)

    #Insert rows
    content_value = "Foundry Local runs models locally."
    embedding_value = "fake_vector_123"
    cursor.execute("INSERT INTO documents (content, embedding) VALUES (?,?)", (content_value, embedding_value) )

    content_value2 = "Microsoft Created Windows"
    embedding_value2 = "fake_vector_345"
    cursor.execute("INSERT INTO documents (content, embedding) VALUES (?,?)", (content_value2, embedding_value2) )
    
    content_value3 = "Toyota is the most reliable car brand."
    embedding_value3 = "fake_vector_678"
    cursor.execute("INSERT INTO documents (content, embedding) VALUES (?,?)", (content_value3, embedding_value3) )



    #Commit Changes
    conn.commit()

    #Query all rows back
    cursor.execute("SELECT * FROM documents")
    rows = cursor.fetchall()
    for lines in rows: print(lines)




if __name__ == "__main__":
    main() 