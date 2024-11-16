import numpy as np
import polars as pl
import polars.selectors as cs
from tqdm import tqdm
from hashlib import md5
from pathlib import Path
from inspect import getsource
from typing import Optional, Union, Callable, List, Dict, Any


def md5_hash(x: str):
    return md5(str(x).encode('utf-8')).hexdigest()


def get_optimal_schema(df: pl.DataFrame, ignore: Optional[List[str]] = None) -> pl.Schema:
    selector = cs.integer()
    if ignore is not None:
        selector = selector & (~cs.by_name(*ignore))
    minmax = pl.concat([
        df.select(selector).min(),
        df.select(selector).max(),
    ], how='vertical')
    floats = df.select(cs.float())
    floats_cols = np.asarray(floats.columns)
    if len(floats_cols) > 0:
        arr = floats.to_numpy()
        floats_cols = floats_cols[np.isclose(arr, arr.round()).all(axis=0)]
        floats = pl.concat([
            df.select(*floats_cols).min(),
            df.select(*floats_cols).max(),
        ], how='vertical')
        minmax = pl.concat([minmax, floats], how='horizontal')
    schema = dict(minmax.schema)
    for dtype in (pl.Int32, pl.UInt32, pl.Int16, pl.UInt16, pl.Int8, pl.UInt8):
        df_dict = (
            minmax
            .cast({cs.all(): dtype}, strict=False)
            .to_dict(as_series=False)
        )
        for k, v in df_dict.items():
            if v[0] is not None and v[1] is not None:
                schema[k] = dtype
    return {cs.float(): pl.Float32, **schema}


class Feature:

    def __init__(
            self,
            name: str,
            callback: Callable,
            dependencies: Optional[List[str]] = None,
            variables: Optional[Dict[str, Any]] = None,
            join_on: Optional[Union[str, List[str]]] = None,
            cache: Optional[bool] = True,
            streaming: Optional[bool] = False,
    ):
        self.name = name
        self.callback = callback
        self.dependencies = dependencies or list()
        self.variables = variables or dict()
        self.join_on = join_on
        self.cache = cache
        self.streaming = streaming

    def hashdict(self) -> str:
        hashdict = self.variables.copy()
        hashdict['__sourcecode__'] = getsource(self.callback)
        return hashdict

    def collect(self, registry: 'FeatureRegistry') -> pl.LazyFrame:
        dependencies_keys = sorted(self.dependencies)

        hashdict = self.hashdict()
        hashdict['__parent__'] = {dep: registry.registry_[dep].hashdict() for dep in dependencies_keys}
        hashsum = md5_hash(hashdict)

        filepath = registry.storage_path / f'{self.name}___{hashsum}.parquet'
    
        if filepath.exists() and self.cache:
            return pl.scan_parquet(filepath, cache=False)

        variables = self.variables.copy()
        if self.join_on is not None:
            variables['join_on'] = self.join_on.copy()

        dependencies = {dep: registry[dep] for dep in dependencies_keys}

        result = self.callback(**dependencies, **variables).collect(streaming=self.streaming)
        schema = get_optimal_schema(result, ignore=self.join_on)
        result = result.cast(schema)

        if not self.cache:
            return result.lazy()
        else:
            result.write_parquet(filepath)
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
        ):
        assert name not in self.registry_, name
        if variables is not None:
            variables = {var: self.variables[var] for var in sorted(variables)}
        def decorator(callback):
            feature_name = name or callback.__name__
            self.registry_[feature_name] = Feature(
                name=feature_name,
                callback=callback,
                dependencies=dependencies,
                variables=variables,
                join_on=join_on,
                cache=cache,
                streaming=streaming,
            )
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
