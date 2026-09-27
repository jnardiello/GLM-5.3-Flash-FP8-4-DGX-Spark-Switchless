import sqlite3, os, tempfile
from migrate import migrate
def test_keeps_data():
    p=os.path.join(tempfile.mkdtemp(),'db.sqlite'); con=sqlite3.connect(p)
    con.execute("CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT, email TEXT)")
    con.execute("INSERT INTO users VALUES (1,'ann','ann@x.io'),(2,'bob','bob@x.io')"); con.commit(); con.close()
    migrate(p); con=sqlite3.connect(p)
    assert con.execute("SELECT name,email,active FROM users ORDER BY id").fetchall()==[('ann','ann@x.io',1),('bob','bob@x.io',1)]
