import subprocess, sys
def test_plain():
    assert subprocess.run([sys.executable,'cli.py','3','4'],capture_output=True,text=True).stdout.strip()=='7'
