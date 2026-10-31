from typing import Protocol


class LiteratureSearch(Protocol):
    def search(self, query: str) -> tuple[str, ...]: ...
