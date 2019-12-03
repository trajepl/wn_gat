import os
from multiprocessing import Pool
from typing import Any, Callable, List


class ParallerParser(object):
    def __init__(
            self,
            total: int,
            cores: int = os.cpu_count(),
            **kwargs
    ) -> None:
        self.total = total
        self.kwargs = kwargs
        self.cores = min(cores, os.cpu_count())
        self.step = max(1, int(self.total / self.cores))

    def run(self) -> List[Any]:
        # print(f'paraller_parse on {self.cores} cpus')
        pool = Pool(processes=self.cores)
        rls = []
        for i in range(0, self.total, self.step):
            rls.append(pool.apply_async(
                func=self.parser,
                args=(i, i+self.step),
                kwds=self.kwargs
            ))
        pool.close()
        pool.join()

        return rls

    # test
    def parser(
            self,
            l: int,
            r: int,
            **kwargs
    ) -> Any:
        print(l, r)


if __name__ == "__main__":
    parser = ParallerParser(total=18)
    parser.run()
