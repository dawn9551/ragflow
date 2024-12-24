import infinity
from infinity.common import ConflictType, InfinityException, SortType
from infinity.index import IndexInfo, IndexType
from infinity.connection_pool import ConnectionPool
from infinity.errors import ErrorCode

infinity_uri = infinity.common.NetworkAddress("43.153.92.239", 23817)
connPool = ConnectionPool(infinity_uri)
inf_conn = connPool.get_conn()
res = inf_conn.show_current_node()
print(f"code: {res.error_code}, status: {res.server_status}")
if res.error_code == ErrorCode.OK and res.server_status == "started":
    print("Infinity is healthy.")
else:
    print("Infinity is not healthy.")
