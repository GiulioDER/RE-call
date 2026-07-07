from recall_mcp.server import build_server


def test_server_builds_with_three_tools():
    mcp = build_server()
    assert mcp.name == "recall_mcp"
    tool_names = {t.name for t in mcp._tool_manager.list_tools()}
    assert {"recall_search", "recall_index", "recall_stats"} <= tool_names
