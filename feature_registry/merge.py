import polars as pl
from typing import Optional, Dict, List, Any


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
        fill_null: Optional[Any] = None,
        column_name: Optional[str] = None,
):
    suffix = '' if suffix is None else (sep + suffix)
    postfix = '' if postfix is None else (sep + postfix)
    if strategy == 'name':
        frames = {
            key: df.with_columns(
                pl.exclude(on).name.suffix(f'{suffix}{key}{postfix}')
            )
            for key, df in frames.items()
        }
    elif strategy == 'column':
        assert column_name is not None
        frames = {
            key: df.with_columns(
                pl.lit(f'{suffix}{key}{postfix}').alias(column_name)
            )
            for key, df in frames.items()
        }
    else:
        raise NotImplementedError(strategy)
    it = iter(frames.keys())
    df = frames[next(it)]
    for key in it:
        df = df.join(frames[key], on=on, how='full', coalesce=True)
    if fill_null is not None:
        df = df.fill_null(fill_null)
    return df


__all__ = [
    'merge_list',
    'merge_dict',
]
