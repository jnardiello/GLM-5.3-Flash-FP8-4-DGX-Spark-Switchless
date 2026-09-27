The SQLite migration in migrate.py corrupts existing rows (tests/test_migrate.py). Fix the migration so old rows keep their data and the new column gets a sensible default.
