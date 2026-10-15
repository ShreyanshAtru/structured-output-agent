"""Shared strict contracts for application data."""
from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
