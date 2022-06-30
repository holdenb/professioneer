from typing import Optional
from dataclasses import dataclass, field
from dataclasses_json import config, dataclass_json


@dataclass_json
@dataclass()
class CraftingPattern:
    """
    Defines the structure for a crafting pattern:
        item: Formatted item name
        category: Item category (i.e. Crafting material)
        materials: Dictionary of materials and their quantities i.e. {"Linen Cloth": 1}
        skill: Dictionary of skill colors and their associated levels i.e {"Orange": 1}
        source: Source of the pattern
        cost: Sum cost of materials relative to market prices (default: 0)
    """
    item: str = field(metadata=config(field_name="Item"))
    category: str = field(metadata=config(field_name="Category"))
    materials: dict[str, int] = field(metadata=config(field_name="Materials"))
    skill: dict[str, str] = field(metadata=config(field_name="Skill"))
    source: str = field(metadata=config(field_name="Source"))
    cost: dict[str, int] = field(init=False)

    def __post_init__(self):
        self.cost = {}


@dataclass_json
@dataclass(frozen=True)
class MarketData:
    """
    Defines the structure of market data:
        market_value: Avg. market value across all postings
            (in copper i.e gold cost * 100s * 100c)
        min_buyout: Avg. minimum buyout across all postings
        quantity: Total amount of items within all postings
        scanned_at: Datetime of AH scan (fmt: 2022-06-14T04:42:29.000Z)
    """
    market_value: Optional[int] = field(metadata=config(field_name="marketValue"))
    min_buyout: Optional[int] = field(metadata=config(field_name="minBuyout"))
    quantity: Optional[int]
    scanned_at: Optional[str] = field(metadata=config(field_name="scannedAt"))


@dataclass_json
@dataclass(frozen=True)
class MarketItem:
    """
    Defines the structure of a market item:
        slug: Server-faction string
        item_id: Unique item ID
        name: Formatted name
        unique_name: Unique name identifier
        time_range: Time between each data point
        data: Optional market data for an item
            Data will be in the format of a list containing MarketData
    """
    slug: str
    item_id: int = field(metadata=config(field_name="itemId"))
    name: str
    unique_name: str = field(metadata=config(field_name="uniqueName"))
    time_range: int = field(metadata=config(field_name="timerange"))
    data: Optional[list[MarketData]]
