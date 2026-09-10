"""Asynchronous reader/writer lock with task ownership and writer preference.

``RWLock`` coordinates access between :mod:`asyncio` tasks in one event loop.
Multiple readers may hold the lock concurrently, while a writer excludes both
readers and other writers. A writer waiting in the queue prevents new readers
from entering, which gives writers preference and avoids an endless stream of
new readers starving a writer.

The lock is reentrant by task: a task may acquire a read lock repeatedly, or a
write lock repeatedly, and must release each acquisition the same number of
times. A task that owns the write lock may also acquire a nested read lock. The
reverse operation, upgrading from read to write in the same task, is rejected
because waiting for other readers while still holding the read lock can
deadlock the lock.

This class is an asyncio synchronization primitive, not a thread
synchronization primitive. Do not share one instance across event loops or use
it to protect synchronous threads.
"""

import asyncio


class RWLock:
    """Task-aware asynchronous reader/writer lock.

    The lock has two independent ownership records: one writer task and a
    per-task reader-depth mapping. The internal :class:`asyncio.Lock` protects
    those records, and the condition variable wakes blocked tasks whenever a
    writer or the final reader releases its ownership.

    Reader admission follows this rule for a task that does not already own
    the lock:

    * wait while a writer is active; and
    * also wait while any writer is queued.

    Consequently, readers already admitted may run concurrently, but readers
    arriving after a writer starts waiting yield to that writer.

    Ownership is associated with the current :class:`asyncio.Task`, not with a
    coroutine object or an arbitrary caller token. Every successful acquire
    must be paired with a release by the same task.
    """

    def __init__(self):
        """Initialize an unlocked reader/writer lock for the current event loop.

        The constructor creates no tasks and does not acquire any lock. The
        instance should be created and used from the event-loop context that
        owns it. Internal state is protected by an asyncio mutex; callers
        should use :attr:`reader_lock` and :attr:`writer_lock` rather than
        accessing the internal records.
        """
        # A mutex protects internal state.
        self._lock = asyncio.Lock()
        # A condition variable to signal when the state changes.
        self._cond = asyncio.Condition(self._lock)

        # Store the task holding the writer lock and the reentrancy depth.
        self._writer_task: asyncio.Task | None = None
        self._writer_depth: int = 0

        # Save the list of tasks currently being read and the reentrant depth of each task:
        # {Task: depth}
        self._reader_tasks: dict[asyncio.Task, int] = {}

        # Number of writer tasks in the queue
        self._waiting_writers: int = 0

    @property
    def reader_lock(self):
        """Return an async context manager for one reader acquisition.

        Each property access creates a lightweight context-manager wrapper. A
        typical use is::

            async with lock.reader_lock:
                value = shared_state.read()

        Entering calls :meth:`acquire_reader`; leaving calls
        :meth:`release_reader`, including when the body raises an exception.
        The wrapper does not suppress exceptions from the protected block.
        """
        return _ReaderLock(self)

    @property
    def writer_lock(self):
        """Return an async context manager for one writer acquisition.

        Each property access creates a lightweight context-manager wrapper. A
        typical use is::

            async with lock.writer_lock:
                shared_state.update(value)

        Entering calls :meth:`acquire_writer`; leaving calls
        :meth:`release_writer`, including when the body raises an exception.
        The wrapper does not suppress exceptions from the protected block.
        """
        return _WriterLock(self)

    async def acquire_reader(self):
        """Acquire one reentrant reader level for the current task.

        A task that already owns a reader level increments its depth without
        waiting. A task that owns the writer lock may also acquire a nested
        reader level immediately; this is useful for shared helper functions
        called from a writer section. A new task waits while a writer is active
        or while any writer is queued, preserving writer preference.

        The method suspends only while waiting for admission. Once admitted, it
        records one reader depth for the current task before returning.

        Raises:
            RuntimeError: If called outside an active ``asyncio.Task``. This
                also occurs when a caller tries to use the method from a plain
                synchronous context.

        Notes:
            A reader acquisition must be released by the same task with
            :meth:`release_reader`. Reader-to-writer upgrade is intentionally
            unsupported; :meth:`acquire_writer` raises instead.
        """
        current = asyncio.current_task()
        if current is None:
            raise RuntimeError("Must be called within an asyncio Task.")

        async with self._lock:
            # Case 1: The current task is already a Writer -> Allow downgrade / acquire read.
            if self._writer_task == current:
                self._reader_tasks[current] = self._reader_tasks.get(current, 0) + 1
                return

            # Case 2: The current task is already a Reader -> Increment reentrance depth.
            if current in self._reader_tasks:
                self._reader_tasks[current] += 1
                return

            # Case 3: New task -> Must wait if a Writer is currently running or pending.
            while self._writer_task is not None or self._waiting_writers > 0:
                await self._cond.wait()

            self._reader_tasks[current] = 1

    async def release_reader(self):
        """Release one reader level held by the current task.

        Reentrant acquisitions are released one level at a time. The task is
        removed from the reader-owner map only after its depth reaches zero.
        When it was the final reader, waiting writers and readers are notified
        so they can re-check their admission conditions.

        Raises:
            RuntimeError: If called outside an active ``asyncio.Task`` or if
                the current task does not hold a reader lock.

        Notes:
            Release exactly once for each successful :meth:`acquire_reader`
            call. A writer task that acquired a nested reader lock must also
            release that reader level before releasing its writer level.
        """
        current = asyncio.current_task()
        if current is None:
            raise RuntimeError("Must be called within an asyncio Task")

        async with self._lock:
            if current not in self._reader_tasks:
                raise RuntimeError("Current task does not hold a reader lock")

            self._reader_tasks[current] -= 1
            if self._reader_tasks[current] == 0:
                del self._reader_tasks[current]
                # If there are no more readers, wake up the waiting tasks.
                if len(self._reader_tasks) == 0:
                    self._cond.notify_all()

    async def acquire_writer(self):
        """Acquire one exclusive, reentrant writer level.

        A task that already owns the writer lock increments the writer depth
        and returns immediately. A new writer waits until there is no active
        writer and no reader task. While waiting, it increments
        ``_waiting_writers`` so new readers yield to it. The counter is reduced
        in a ``finally`` block, including if the waiting task is cancelled.

        A task holding a reader lock cannot upgrade to a writer lock. The
        operation raises immediately rather than waiting while retaining its
        reader ownership and risking deadlock. To change from read access to
        write access, release every reader level first and then acquire the
        writer lock.

        Raises:
            RuntimeError: If called outside an active ``asyncio.Task``, if the
                current task already holds a reader lock, or if the lock's
                ownership state does not permit the requested acquisition.

        Notes:
            Each successful acquisition must be paired with
            :meth:`release_writer` by the same task. A writer may acquire a
            nested reader lock, but that nested reader acquisition has its own
            release obligation.
        """
        current = asyncio.current_task()
        if current is None:
            raise RuntimeError("Must be called within an asyncio Task.")

        async with self._lock:
            # Case 1: The current task is already a Writer -> Increase reentrance depth.
            if self._writer_task == current:
                self._writer_depth += 1
                return

            # Deadlock warning: If a task currently acting as a
            # Reader requests Writer status while other Readers are present.
            if current in self._reader_tasks:
                raise RuntimeError(
                    "It is not possible to upgrade from Reader to Writer within the same task."
                )

            # Case 2: New writer -> Wait until all active readers and writers have finished.
            self._waiting_writers += 1
            try:
                while self._writer_task is not None or len(self._reader_tasks) > 0:
                    await self._cond.wait()
                self._writer_task = current
                self._writer_depth = 1
            finally:
                self._waiting_writers -= 1

    async def release_writer(self):
        """Release one writer level held by the current task.

        Reentrant writer acquisitions decrease ``_writer_depth`` one level at
        a time. The writer ownership record is cleared only at depth zero;
        then all waiters are notified so the next eligible writer or group of
        readers can proceed.

        Raises:
            RuntimeError: If called outside an active ``asyncio.Task`` or if
                the current task is not the writer owner.

        Notes:
            If the writer task acquired nested reader levels, release those
            reader levels before releasing the final writer level. The lock
            intentionally does not infer or repair mismatched release calls.
        """
        current = asyncio.current_task()
        if current is None:
            raise RuntimeError("Must be called within an asyncio Task.")

        async with self._lock:
            if self._writer_task != current:
                raise RuntimeError("The current task does not hold the writer lock.")

            self._writer_depth -= 1
            if self._writer_depth == 0:
                self._writer_task = None
                # Wake up other Readers or Writers in the queue
                self._cond.notify_all()


class _ReaderLock:
    """Async context-manager adapter for one :class:`RWLock` read level."""

    def __init__(self, rwlock: RWLock):
        """Bind the adapter to an existing reader/writer lock.

        Args:
            rwlock: Lock whose reader acquisition and release methods should
                be called by this context manager.
        """
        self._rwlock = rwlock

    async def __aenter__(self):
        """Acquire the underlying reader lock and return this adapter."""
        await self._rwlock.acquire_reader()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Release the reader lock without suppressing body exceptions."""
        await self._rwlock.release_reader()


class _WriterLock:
    """Async context-manager adapter for one :class:`RWLock` write level."""

    def __init__(self, rwlock: RWLock):
        """Bind the adapter to an existing reader/writer lock.

        Args:
            rwlock: Lock whose writer acquisition and release methods should
                be called by this context manager.
        """
        self._rwlock = rwlock

    async def __aenter__(self):
        """Acquire the underlying writer lock and return this adapter."""
        await self._rwlock.acquire_writer()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Release the writer lock without suppressing body exceptions."""
        await self._rwlock.release_writer()
