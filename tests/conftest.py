"""Shared pytest fixtures."""
from __future__ import annotations

import os
import pytest
import pandas as pd

# Point MLflow at a local temp dir so tests don't need Databricks
os.environ.setdefault("MLFLOW_TRACKING_URI", "./mlruns_test")
os.environ.setdefault("QC_ENV", "local")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    """Small DataFrame with intentional QC issues."""
    return pd.DataFrame([
        # id,  name,              address_line1,       city,          state,      zip_code,     email,                  phone
        {"id": "1", "name": "Alice Smith",    "address_line1": "123 Main St",      "city": "New York",    "state": "NY", "zip_code": "10001", "email": "alice@example.com",  "phone": "212-555-0100"},
        {"id": "2", "name": "Bob Jones",      "address_line1": "456 Oak Ave",      "city": "Los Angles",  "state": "CA", "zip_code": "90001", "email": "bob@example.com",    "phone": "310-555-0200"},  # city typo
        {"id": "3", "name": "Carol White",    "address_line1": "789 Elm Blvd",     "city": "Chcago",      "state": "IL", "zip_code": "60601", "email": "carol@example.com",  "phone": "312-555-0300"},  # city typo
        {"id": "4", "name": "Dave Brown",     "address_line1": "321 Pine Rd",      "city": "Houston",     "state": "TX", "zip_code": "7700",  "email": "dave@example.com",   "phone": "713-555-0400"},  # zip typo
        {"id": "5", "name": "Eve Davis",      "address_line1": "654 Cedar Ln",     "city": "Pheonix",     "state": "AZ", "zip_code": "85001", "email": "not-an-email",        "phone": "602-555-0500"},  # city typo + email format
        {"id": "6", "name": None,             "address_line1": "111 Maple Dr",     "city": "Seattle",     "state": "WA", "zip_code": "98101", "email": "f@g.com",            "phone": "206-555-0600"},  # null name
        {"id": "7", "name": "Grace Lee",      "address_line1": "222 Birch Way",    "city": "Denver",      "state": "Colorad", "zip_code": "80201", "email": "grace@example.com", "phone": "720-555-0700"},  # state typo
        {"id": "8", "name": "Henry Wilson",   "address_line1": "333 Walnut Ct",    "city": "Boston",      "state": "MA", "zip_code": "02101", "email": "henry@example.com",  "phone": "617-555-0800"},
        {"id": "9", "name": "Iris Clark",     "address_line1": "444 Spruce Pl",    "city": "Miami",       "state": "FL", "zip_code": "33101", "email": "iris@example.com",   "phone": "305-555-0900"},
        {"id": "10","name": "Jack Turner",    "address_line1": "555 Ash Ter",      "city": "Atlnata",     "state": "GA", "zip_code": "30301", "email": "jack@example.com",   "phone": "404-555-1000"},  # city typo
    ])


@pytest.fixture
def sample_csv_path(tmp_path, sample_df) -> str:
    path = str(tmp_path / "sample_customers.csv")
    sample_df.to_csv(path, index=False)
    return path


@pytest.fixture
def mock_config():
    from configs.settings import AppConfig
    return AppConfig()


@pytest.fixture
def mock_bus():
    from messaging.message_bus import MessageBus
    return MessageBus()
