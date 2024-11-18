import polars as pl
from typing import Optional, Dict, List


def merge_list(frames: List[pl.LazyFrame], on: List[str]) -> pl.LazyFrame:
    df1 = frames[0]
    for df2 in frames[1:]:
        df1 = df1.join(df2, on=on, how='full', coalesce=True)
    return df1


def merge_dict(
        frames: Dict[int, pl.LazyFrame],
        on: List[str],
        strategy: Optional[str] = None,
        sep: Optional[str] = '/',
        suffix: Optional[str] = None,
        postfix: Optional[str] = None,
        fill_null: Optional[float] = None,
        column_name: Optional[str] = None,
):
    postfix = '' if postfix is None else postfix
    if strategy == 'name':
        suffix = '' if suffix is None else (sep + suffix)
        df = merge_list([
            df.rename({
                col: f'{col}{suffix}{sep}{key}{postfix}'
                for col in df.collect_schema().names()
                if col not in on
            })
            for key, df in frames.items()
        ], on=on)
    elif strategy == 'column':
        assert column_name is not None
        df = pl.concat([
            df.with_columns(
                pl.lit(f'{key}{postfix}').alias(column_name)
            )
            for key, df in frames.items()
        ])
    else:
        raise NotImplementedError(strategy)
    if fill_null is not None:
        df = df.fill_null(fill_null)
    return df


__all__ = [
    'merge_list',
    'merge_dict',
]
