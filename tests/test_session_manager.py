import asyncio
from threading import Event

import pytest

from server.session_manager import MaxSessionError, SessionManager


class _IsolatedSessionManager(SessionManager):
    _instance = None


def _fresh_manager(builder):
    _IsolatedSessionManager._instance = None
    manager = _IsolatedSessionManager()
    manager.build_session_fn = builder
    manager.max_session = 5
    manager._build_lock = asyncio.Lock()
    return manager


def test_reconnect_cannot_start_a_second_session_build():
    started = Event()
    release = Event()
    avatar = object()

    def builder(_sessionid, _params):
        started.set()
        assert release.wait(2)
        return avatar

    async def scenario():
        manager = _fresh_manager(builder)
        first = asyncio.create_task(manager.create_session({}))
        while not started.is_set():
            await asyncio.sleep(0.01)

        with pytest.raises(MaxSessionError, match="still initializing"):
            await manager.create_session({})

        release.set()
        sessionid = await first
        assert manager.get_session(sessionid) is avatar
        assert len(manager.sessions) == 1

    asyncio.run(scenario())


def test_failed_session_build_releases_its_reservation():
    def builder(_sessionid, _params):
        raise RuntimeError("engine load failed")

    async def scenario():
        manager = _fresh_manager(builder)
        with pytest.raises(RuntimeError, match="engine load failed"):
            await manager.create_session({})
        assert manager.sessions == {}
        assert not manager._build_lock.locked()

    asyncio.run(scenario())
