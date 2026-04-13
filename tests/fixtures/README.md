# Test Fixtures

Place sample CSV/Parquet files here for integration and e2e tests.

The `conftest.py` `sample_df` fixture generates an in-memory DataFrame with
intentional QC issues. The `sample_csv_path` fixture writes it to a temp CSV.
