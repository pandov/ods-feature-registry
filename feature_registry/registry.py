import random
import numpy as np
import polars as pl
import polars.selectors as cs
from tqdm import tqdm
from hashlib import md5
from pathlib import Path
from inspect import getsource
from typing import Optional, Union, Callable, Tuple, List, Dict, Any


INTEGERS = (pl.Int32, pl.UInt32, pl.Int16, pl.UInt16, pl.Int8, pl.UInt8)


def get_optimal_schema(df: pl.DataFrame, ignore: Optional[List[str]] = None) -> pl.Schema:
    selector = cs.integer()
    if ignore is not None:
        selector = selector & (~cs.by_name(*ignore))
    minmax = pl.concat([
        df.select(selector).min(),
        df.select(selector).max(),
    ], how='vertical')
    # floats = df.select(cs.float()).cast(pl.Float32)
    # floats_cols = np.asarray(floats.columns)
    # if len(floats_cols) > 0:
    #     arr = floats.to_numpy()
    #     isclose = np.isclose(arr, arr.round(), rtol=1e-2, atol=1e-2, equal_nan=True).all(axis=0)
    #     floats_cols = floats_cols[isclose]
    #     floats = pl.concat([
    #         df.select(*floats_cols).min(),
    #         df.select(*floats_cols).max(),
    #     ], how='vertical')
    #     minmax = pl.concat([minmax, floats], how='horizontal')
    schema = dict(minmax.schema)
    for dtype in INTEGERS:
        df_dict = minmax.cast(dtype, strict=False).to_dict(as_series=False)
        for k, v in df_dict.items():
            if v[0] is not None and v[1] is not None:
                # if v[0] == 0 and v[1] == 1 and schema[k] in INTEGERS:
                #     dtype = pl.Boolean
                schema[k] = dtype
    return {cs.float(): pl.Float32, **schema}


def random_seed(random_state: Optional[int] = 42):
    random.seed(random_state)
    np.random.seed(random_state)


def md5_hash(x: str):
    return md5(str(x).encode('utf-8')).hexdigest()


class FeatureStore:

    def __init__(
            self,
            name: str,
            callback: Callable,
            dependencies: List[str],
            variables: Dict[str, Any],
            join_on: Union[str, List[str]],
            cache: bool,
            streaming: bool,
            hashing: bool,
    ):
        self.name = name
        self.callback = callback
        self.dependencies = sorted(dependencies or list())
        self.variables = variables or dict()
        self.join_on = join_on
        self.cache = cache
        self.streaming = streaming
        self.hashing = hashing

    def hashdict(self, registry: 'FeatureRegistry') -> Dict[str, Any]:
        hashdict = self.variables.copy()
        source = getsource(self.callback)
        if source.startswith('@'):
            source = '\n'.join(source.split('\n')[1:])
        hashdict['__sourcecode__'] = source
        hashdict['__parent__'] = {dep: registry.registry_[dep].hashdict(registry) for dep in self.dependencies}
        return hashdict

    def get_write_meta(self, registry: 'FeatureRegistry') -> Tuple[str, str]:
        stem = self.name
        if self.hashing:
            hashsum = md5_hash(self.hashdict(registry))
            stem += f'___{hashsum}'
        if self.join_on is None or 'date' not in self.join_on:
            stem += '.parquet'
            partition_by = None
        else:
            partition_by = 'date'
        return stem, partition_by

    def collect(self, registry: 'FeatureRegistry') -> pl.LazyFrame:
        random_seed()

        stem, partition_by = self.get_write_meta(registry)
        filepath = registry.storage_path / stem
    
        if filepath.exists() and self.cache:
            return pl.scan_parquet(filepath, cache=False)

        variables = self.variables.copy()
        if self.join_on is not None:
            variables['join_on'] = self.join_on.copy()

        dependencies = {dep: registry[dep] for dep in self.dependencies}

        result = self.callback(**dependencies, **variables).collect(streaming=self.streaming)
        schema = get_optimal_schema(result, ignore=self.join_on)
        result = result.cast(schema)

        if not self.cache:
            return result.lazy()
        else:
            result.write_parquet(filepath, partition_by=partition_by)
            return pl.scan_parquet(filepath, cache=False)


class FeatureRegistry:

    def __init__(self, storage_path: str, **variables):
        self.registry_ = {}
        self.variables = variables
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True, parents=True)

    def __repr__(self) -> str:
        return str(list(self.registry_.keys()))

    def __getitem__(self, index: Union[str, int]) -> pl.LazyFrame:
        if isinstance(index, int):
            index = self.features[index]
        return self.get(index)

    def __len__(self) -> int:
        return len(self.registry_)

    @property
    def features(self) -> List[str]:
        return list(self.registry_.keys())

    def get(self, name: str):
        assert name in self.registry_, name
        return self.registry_[name].collect(self)

    def add(
            self,
            name: str = None,
            dependencies: Optional[List[str]] = None,
            variables: Optional[List[str]] = None,
            join_on: Optional[Union[str, List[str]]] = None,
            cache: Optional[bool] = True,
            streaming: Optional[bool] = False,
            hashing: Optional[bool] = True,
        ):
        assert name not in self.registry_, name
        if variables is not None:
            variables = {var: self.variables[var] for var in sorted(variables)}
        def decorator(callback):
            feature_name = name or callback.__name__
            self.registry_[feature_name] = FeatureStore(
                name=feature_name,
                callback=callback,
                dependencies=dependencies,
                variables=variables,
                join_on=join_on,
                cache=cache,
                streaming=streaming,
                hashing=hashing,
            )
            return callback
        return decorator

    def collect(self):
        pbar = tqdm(self.registry_.items(), desc='Collecting')
        for name, feature in pbar:
            pbar.set_postfix({'feature': name})
            if feature.cache:
                feature.collect(self)

    def join(
            self,
            df: pl.LazyFrame,
            feature_fn: Optional[Callable] = None,
            selector: Optional[cs._selector_proxy_] = None,
    ) -> pl.LazyFrame:
        dates = df.select('date').unique().collect().to_series()
        filter_expr = pl.col('date').is_between(dates.min(), dates.max())

        features = self.features
        if feature_fn is not None:
            features = [name for name in features if feature_fn(name)]
        for name in features:
            feature = self.registry_[name]
            on = feature.join_on
            if on is None:
                continue
            assert 'date' in on, name
            other = self.get(name).filter(filter_expr)
            if selector is not None:
                other = other.select(selector)
            df = df.join(other, on=on, how='left')
        return df


__all__ = ['FeatureRegistry']
