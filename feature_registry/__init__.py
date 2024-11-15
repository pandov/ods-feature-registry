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
    if ignore is not None:
        df = df.drop(ignore)
    minmax = pl.concat([
        df.select(cs.integer()).min(),
        df.select(cs.integer()).max(),
    ])
    schema = dict(minmax.schema)
    for dtype in (pl.Int32, pl.UInt32, pl.Int16, pl.UInt16, pl.Int8, pl.UInt8):
        df_dict = (
            minmax
            .cast({cs.integer(): dtype}, strict=False)
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
    ):
        self.name = name
        self.callback = callback
        self.dependencies = dependencies or list()
        self.variables = variables or dict()
        self.join_on = join_on
        self.cache = cache

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

        result = self.callback(**dependencies, **variables).collect()
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
            )
        return decorator

    def collect(self):
        pbar = tqdm(self.registry_.items(), desc='Collecting')
        for name, feature in pbar:
            pbar.set_postfix({'feature': name})
            if feature.cache:
                feature.collect(self)

    def join(self, df: pl.LazyFrame, features: Optional[List[str]] = None, filter_expr: Optional[pl.Expr] = None) -> pl.LazyFrame:
        if filter_expr is not None:
            df = df.filter(filter_expr)
        if features is None:
            feature = self.features
        for name in features:
            feature = self.registry_[name]
            on = feature.join_on
            if on is None:
                continue
            assert 'date' in on, name
            other = self.get(name)
            if filter_expr is not None:
                other = other.filter(filter_expr)
            df = df.join(other, on=on, how='left')
        return df


__all__ = ['FeatureRegistry']
