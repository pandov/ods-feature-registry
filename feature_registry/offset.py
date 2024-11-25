import polars as pl
from datetime import date
from typing import Optional, List


def offset_dates(df: pl.LazyFrame, offsets: List[int], min_date: date, include_offset_column: Optional[bool] = False) -> pl.LazyFrame:
    return (
        pl.concat([
            df.with_columns(
                [pl.col('date').dt.offset_by(f'{-offset}d')] + [pl.lit(offset).alias('offset')] if include_offset_column else []
            )
            for offset in offsets
        ])
        .filter(pl.col('date') >= min_date)
    )


__all__ = ['offset_dates']
