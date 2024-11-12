import polars as pl
from tqdm import tqdm
from hashlib import md5
from pathlib import Path
from inspect import getsource
from typing import Optional, Union, Callable, List, Dict, Any


def md5_hash(x: str):
    return md5(str(x).encode('utf-8')).hexdigest()


class Feature:

    def __init__(
            self,
            name: str,
            callback: Callable,
            dependencies: Optional[List[str]] = None,
            variables: Optional[Dict[str, Any]] = None,
            join_on: Optional[Union[str, List[str]]] = None,
            cache: Optional[bool] = True,
            scan_params: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.callback = callback
        self.dependencies = dependencies or list()
        self.variables = variables or dict()
        self.join_on = join_on
        self.cache = cache
        if scan_params is None:
            scan_params = dict()
        self.scan_params = scan_params

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
            return pl.scan_parquet(filepath, **self.scan_params)

        variables = self.variables.copy()
        if self.join_on is not None:
            variables['join_on'] = self.join_on.copy()

        dependencies = {dep: registry[dep] for dep in dependencies_keys}
    
        result = self.callback(**dependencies, **variables)

        if not self.cache:
            return result
        else:
            result.collect().write_parquet(filepath)
            return pl.scan_parquet(filepath, **self.scan_params)


class FeatureRegistry:

    def __init__(self, storage_path: str, scan_params: Dict[str, Any] = None, **variables):
        self.registry_ = {}
        self.variables = variables
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True, parents=True)
        self.scan_params = scan_params

    def __repr__(self) -> str:
        return str(list(self.registry_.keys()))

    def __getitem__(self, name: str) -> pl.LazyFrame:
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
                scan_params=self.scan_params,
            )
        return decorator

    def collect(self):
        pbar = tqdm(self.registry_.items(), desc='Collecting')
        for name, feature in pbar:
            pbar.set_postfix({'feature': name})
            if feature.cache:
                feature.collect(self)

    def join(self, df: pl.LazyFrame) -> pl.LazyFrame:
        for name, feature in self.registry_.items():
            on = feature.join_on
            if on is None:
                continue
            assert 'date' in on, name
            df = df.join(
                other=self[name],
                on=on,
                how='left',
            )
        return df


__all__ = ['FeatureRegistry']
