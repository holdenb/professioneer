from typing import Optional
from dataclasses import dataclass, field
from dataclasses_json import config, dataclass_json

@dataclass_json
@dataclass(frozen=True)
class Profession:
    item: str = field(metadata=config(field_name="Item"))
    category: str = field(metadata=config(field_name="Category"))
    materials: dict[str, int] = field(metadata=config(field_name="Materials"))
    skill: dict[str, str] = field(metadata=config(field_name="Skill"))
    source: str = field(metadata=config(field_name="Source"))

@dataclass_json
@dataclass(frozen=True)
class MarketData:
    market_value: Optional[int] = field(metadata=config(field_name="marketValue"))
    min_buyout: Optional[int] = field(metadata=config(field_name="minBuyout"))
    quantity: Optional[int]
    scanned_at: Optional[str] = field(metadata=config(field_name="scannedAt"))

@dataclass_json
@dataclass(frozen=True)
class MarketItem:
    slug: str
    item_id: int = field(metadata=config(field_name="itemId"))
    name: str
    unique_name: str = field(metadata=config(field_name="uniqueName"))
    time_range: int = field(metadata=config(field_name="timerange"))
    data: Optional[list[MarketData]]
