from contextlib import contextmanager

import psycopg
from recall.store import PgVectorStore
from recall.types import Chunk
from tests.conftest import TEST_DSN, requires_db
from tests import test_store_pool

pooled_store = test_store_pool.pooled_store
_vec = test_store_pool._vec


@requires_db
def test_lost_commit_ack_does_not_replay_over_a_newer_write(pooled_store, monkeypatch):
    """DAT-001: real COMMIT succeeds, acknowledgement fails, newer write must survive.

    Red baseline e5f7892b8: PgVectorStore._with_retry_pooled replays the first
    upsert and the text assertion sees 'older' instead of 'newer'. Only the
    acknowledgement transport failure is injected, after the real DB commit.
    """
    store = pooled_store
    store.upsert([Chunk('same', 's.md', 'initial')], [_vec(0)])
    borrow = store._pool.connection
    lost = False

    class Connection:
        def __init__(self, connection):
            self.connection = connection

        def __getattr__(self, name):
            return getattr(self.connection, name)

        @contextmanager
        def transaction(self, *args, **kwargs):
            nonlocal lost
            outer = self.connection.info.transaction_status == psycopg.pq.TransactionStatus.IDLE
            with self.connection.transaction(*args, **kwargs) as tx:
                yield tx
            if outer and not lost:
                lost = True
                self.connection.close()
                with psycopg.connect(TEST_DSN, autocommit=True) as writer:
                    writer.execute(
                        f'UPDATE {store.table} SET text = %s WHERE tenant_id = %s AND id = %s',
                        ('newer', store.tenant, 'same'),
                    )
                raise psycopg.OperationalError('commit acknowledgement lost')

    @contextmanager
    def connection(*args, **kwargs):
        with borrow(*args, **kwargs) as conn:
            yield Connection(conn)

    monkeypatch.setattr(store._pool, 'connection', connection)
    error = None
    try:
        store.upsert([Chunk('same', 's.md', 'older')], [_vec(0)])
    except psycopg.OperationalError as exc:
        error = exc
    with psycopg.connect(TEST_DSN, autocommit=True) as reader:
        text = reader.execute(
            f'SELECT text FROM {store.table} WHERE tenant_id = %s AND id = %s',
            (store.tenant, 'same'),
        ).fetchone()[0]
    assert text == 'newer', 'an indeterminate COMMIT replay overwrote the newer write'
    assert lost, 'the transport failure was never injected'
    assert error is not None, 'the indeterminate outcome was swallowed'


@requires_db
def test_idle_disconnect_is_recovered_before_the_operation_runs(pooled_store, monkeypatch):
    """DAT-001: checkout recovery must not replay the caller's operation.

    Red on e5f7892b8: count() enters the operation twice after an idle socket
    is terminated. The checkout check must replace it before the first attempt.
    """
    from recall import store as module

    store = PgVectorStore(TEST_DSN, dim=8, table=pooled_store.table, pool_size=1)
    try:
        with store._pool.connection() as connection:
            backend_pid = connection.info.backend_pid
        with psycopg.connect(TEST_DSN, autocommit=True) as admin:
            assert admin.execute('SELECT pg_terminate_backend(%s)', (backend_pid,)).fetchone()[0]

        observed = module._observed_db_call
        attempts = 0

        def operation(connection, callback):
            nonlocal attempts
            attempts += 1
            return observed(connection, callback)

        monkeypatch.setattr(module, '_observed_db_call', operation)
        assert store.count() == 0
        assert attempts == 1, 'idle-connection recovery replayed the caller operation'
    finally:
        store.close()
