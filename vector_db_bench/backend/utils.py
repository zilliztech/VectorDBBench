import time
from multiprocessing.shared_memory import SharedMemory
from contextlib import ContextDecorator
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd


def numerize(n) -> str:
    """display positive number in for readability

    Examples:
        >>> numerize(1_000)
        '1K'
        >>> numerize(1_000_000_000)
        '1B'
    """
    sufix2upbound = {
        "EMPTY": 1e3,
        "K": 1e6,
        "M": 1e9,
        "B": 1e12,
        "END": float('inf'),
    }

    display_n, sufix = n, ""
    for s, base in sufix2upbound.items():
        # number >= 1000B will alway has sufix 'B'
        if s == "END":
            display_n = int(n/1e9)
            sufix = "B"
            break

        if n < base:
            sufix = "" if s == "EMPTY" else s
            display_n = int(n/(base/1e3))
            break
    return f"{display_n}{sufix}"


@dataclass
class Timer(ContextDecorator):
    """simple timer """

    text: str = "elapsed time: {:0.4f}s"
    name: str | None = None
    logger: Callable[[str], None] = print
    _start_time: float | None = field(default=None, init=False, repr=False)

    def start(self) -> None:
        self._start_time = time.perf_counter()

    def stop(self) -> float:
        # Calculate elapsed time
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        # Report elapsed time
        if self.logger:
            self.logger(f"{self.name}: {self.text.format(elapsed_time)}")

        return elapsed_time

    def __enter__(self) -> "Timer":
        """Start a new timer as a context manager"""
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Stop the context manager timer"""
        self.stop()


class SharedNumpyArray:
    ''' Wraps a numpy array so that it can be shared quickly among processes,
    avoiding unnecessary copying and (de)serializing.
    '''
    def __init__(self, array: np.ndarray):
        '''
        Creates the shared memory and copies the array therein
        '''
        # create the shared memory location of the same size of the array
        self._shared = SharedMemory(create=True, size=array.nbytes)

        # save data type and shape, necessary to read the data correctly
        self._dtype, self._shape = array.dtype, array.shape

        # create a new numpy array that uses the shared memory we created.
        # at first, it is filled with zeros
        res = np.ndarray(
            self._shape, dtype=self._dtype, buffer=self._shared.buf
        )

        # copy data from the array to the shared memory. numpy will
        # take care of copying everything in the correct format
        res[:] = array[:]

    def read(self) -> np.ndarray:
        '''Reads the array from the shared memory without unnecessary copying. '''
        # simply create an array of the correct shape and type,
        # using the shared memory location we created earlier
        return np.ndarray(self._shape, self._dtype, buffer=self._shared.buf)

    def unlink(self) -> None:
        ''' Releases the allocated memory. Call when finished using the data,
        or when the data was copied somewhere else.
        '''
        self._shared.close()
        self._shared.unlink()

class SharedDataFrame:
    ''' Wraps a pandas dataframe so that it can be shared quickly
    among processes, avoiding unnecessary copying and (de)serializing.
    '''
    def __init__(self, df):
        ''' Creates the shared memory and copies the dataframe therein '''
        self._values = SharedNumpyArray(df.values)
        self._index = df.index
        self._columns = df.columns

    def read(self):
        ''' Reads the dataframe from the shared memory without unnecessary copying. '''
        return pd.DataFrame(
            self._values.read(),
            index=self._index,
            columns=self._columns
        )

    def unlink(self):
        '''
        Releases the allocated memory. Call when finished using the data,
        or when the data was copied somewhere else.
        '''
        self._values.unlink()
