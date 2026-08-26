from __future__ import annotations

import abc


class BaseScheduler(abc.ABC):
    """Base class for schedulers that manage requests to the engine."""
    
    @abc.abstractmethod
    def add_request(self, request) -> None: ...
    
    @abc.abstractmethod
    def abort(self, request_id: str) -> bool:
        """Return True if the request existed.
        Must be safe to call at any point in the lifecycle, including mid-step.
        """
        
    @abc.abstractmethod
    def schedule(self) -> None: 
        """Produce the next step, or None when there is nothing runnable."""
        
    @abc.abstractmethod
    def update_from_output(self, output) -> None: 
        """Update the scheduler state from the output of a step."""
        
    @property
    @abc.abstractmethod
    def has_unfinished(self) -> bool: ...