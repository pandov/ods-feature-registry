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
    df_max = df.select(cs.integer()).max()
    schema = dict(df_max.schema)
    for dtype in (pl.UInt32, pl.UInt16, pl.UInt8):
        df_dict = (
            df_max
            .cast({cs.integer(): dtype}, strict=False)
            .to_dict(as_series=False)
        )
        for k, v in df_dict.items():
            if v[0] is not None:
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

    def collect(self, registry: 'FeatureRegistry', **scan_params) -> pl.LazyFrame:
        dependencies_keys = sorted(self.dependencies)

        hashdict = self.hashdict()
        hashdict['__parent__'] = {dep: registry.registry_[dep].hashdict() for dep in dependencies_keys}
        hashsum = md5_hash(hashdict)

        filepath = registry.storage_path / f'{self.name}___{hashsum}.parquet'
    
        if filepath.exists() and self.cache:
            return pl.scan_parquet(filepath, **scan_params)

        variables = self.variables.copy()
        if self.join_on is not None:
            variables['join_on'] = self.join_on.copy()

        dependencies = {dep: registry[dep] for dep in dependencies_keys}
    
        result = self.callback(**dependencies, **variables)

        if not self.cache:
            return result
        else:
            result = result.collect()
            print(result.schema)
            result = result.cast(get_optimal_schema(result, ignore=self.join_on))
            print(result.schema)
            result.write_parquet(filepath)
            return pl.scan_parquet(filepath, **scan_params)


class FeatureRegistry:

    def __init__(self, storage_path: str, **variables):
        self.registry_ = {}
        self.variables = variables
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True, parents=True)

    def __repr__(self) -> str:
        return str(list(self.registry_.keys()))

    def __getitem__(self, name: str) -> pl.LazyFrame:
        return self.get(name)

    def get(self, name: str, *args, **kwargs):
        assert name in self.registry_, name
        return self.registry_[name].collect(self, *args, **kwargs)

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
                feature.collect(self, cache=False)

    def join(self, df: pl.LazyFrame) -> pl.LazyFrame:
        for name, feature in self.registry_.items():
            on = feature.join_on
            if on is None:
                continue
            assert 'date' in on, name
            df = df.join(
                other=self.get(name, cache=False),
                on=on,
                how='left',
            )
        return df


__all__ = ['FeatureRegistry']
