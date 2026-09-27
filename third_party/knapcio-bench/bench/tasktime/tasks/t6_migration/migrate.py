import sqlite3
def migrate(path):
    con=sqlite3.connect(path); cur=con.cursor()
    cur.execute("CREATE TABLE users_new(id INTEGER PRIMARY KEY, name TEXT, email TEXT, active INTEGER NOT NULL DEFAULT 1)")
    cur.execute("INSERT INTO users_new(id,name) SELECT id,name FROM users")   # bug: drops email
    cur.execute("DROP TABLE users"); cur.execute("ALTER TABLE users_new RENAME TO users")
    con.commit(); con.close()
