import polars as pl
import polars.selectors as cs
from datetime import date, timedelta
from typing import Optional, Dict, List

from feature_registry.merge import merge_dict


def aggregate_daily_features(
        interactions: pl.LazyFrame,
        by: List[str],
        agg: List[pl.Expr],
) -> pl.LazyFrame:
    return (
        interactions
        .group_by(*by)
        .agg(*agg)
        .collect()
        .lazy()
    )


def filter_last_days(df: pl.LazyFrame, by: List[str], dt: date, window: int) -> pl.LazyFrame:
    assert 'date' not in by, str(by)
    return df.filter(pl.col('date').is_between(dt - timedelta(days=window), dt, closed='left'))  # exclude `dt`


def filter_last_interactions(df: pl.LazyFrame, by: List[str], dt: date, window: int) -> pl.LazyFrame:
    assert 'date' not in by, str(by)
    return (
        df
        .filter(pl.col('date') < dt)  # exclude `dt`
        .with_columns(
            pl.lit(1).alias('index'),
        )
        .with_columns(
            pl.col('index').cum_sum(reverse=True).over(by, order_by='date'),
        )
        .filter(pl.col('index') <= window)
        .drop('index')
    )


def cumulate_daily_features(
        aggregated: pl.LazyFrame,
        by: List[str],
        agg: List[pl.Expr],
        dates: List[date],
        days: List[int],
        strict: Optional[bool] = True,
) -> Dict[int, pl.LazyFrame]:
    selector = [cs.by_name(*by), cs.exclude(*by)]
    by = [x for x in by if x != 'date']
    filter_last_fn = strict and filter_last_days or filter_last_interactions
    return {
        window: pl.concat([
            filter_last_fn(aggregated, by=by, dt=dt, window=window)
            .group_by(*by)
            .agg(agg)
            .with_columns(
                pl.lit(dt).alias('date'),
            )
            .select(*selector)
            for dt in dates
        ])
        for window in sorted(days)
    }


def calculate_daily_features(
        interactions: pl.LazyFrame,
        by: List[str],
        agg: List[pl.Expr],
        cum: List[pl.Expr],
        dates: List[date],
        days: List[int],
        sep: Optional[str] = '/',
        strict: Optional[bool] = True,
        fill_null: Optional[float] = 0,
):
    aggregated = aggregate_daily_features(interactions, by=by, agg=agg)
    cumulated = cumulate_daily_features(aggregated, by=by, agg=cum, dates=dates, days=days, strict=strict)
    suffix = '_'.join([x for x in by if x != 'date']).replace('_id', '')
    return merge_dict(
        frames=cumulated,
        on=by,
        strategy='name',
        suffix=suffix,
        sep=sep,
        fill_null=fill_null,
        postfix='d' if strict else 'i',
    )


__all__ = [
    'aggregate_daily_features',
    'filter_last_days',
    'filter_last_interactions',
    'cumulate_daily_features',
    'calculate_daily_features',
]
